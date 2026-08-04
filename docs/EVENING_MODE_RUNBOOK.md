# Auto EVENING mode - sleep-in runbook

Purpose: get home at 23:30, sleep past the 06:30 open, and still wake to a
phone that guards your positions and a desk that already did the morning's
work.

## What EVENING does (vs DESK / AWAY)

- **Same discovery, zero recommendations.** Trading logic is identical to
  DESK: auto-populate picks stage silently in the Alert Center for chart
  approval. Nothing self-applies and nothing pings the desk while the mode is
  on. Flip the mode off (to DESK or OFF) when you sit down to resume live
  recommendations.
- **Early Master AVWAP run.** The swing scan gets an extra slot at open+30
  (07:00 on a normal session) instead of waiting for the usual open+60, so
  the best-D1 ranking is current when you arrive.
- **Strength persistence checks at 07:00 / 07:15 / 07:30.** Every staged
  intraday pick is snapshotted against its HOD/LOD. A name still pressing its
  extreme at 07:30 is HELD; one that slipped is FADED and explicitly not
  recommended.
- **Morning briefing.** After the 07:30 check the briefing finalizes: market
  environment, best D1 swing setups per side (by expected R), held vs faded
  intraday picks, and any overnight price alerts. It is written to
  `evening_briefing.txt` next to `autopilot_today.txt` on the shared Drive,
  folded into the hourly phone report (EVENING publishes hourly from 07:00
  like AWAY), echoed in the Alert Center, and announced with a normal-priority
  push.
- **Price alerts push at urgent priority** (see below) so a level cross can
  wake you.

## Night-before checklist

1. Leave the GUI running and IB/TWS logged in (same as an Away day).
2. Click the Auto Mode button in the status bar until it reads
   `Auto: EVENING` (cycle: OFF -> DESK -> AWAY -> EVENING -> OFF).
3. Focus -> Phone Price Alerts (or Research -> Price Alerts for the advanced
   view): enter each position, and SPY if you want an index tripwire, with an
   *Alert Above* and/or *Alert Below* level armed.
4. Hit **Test Push** and confirm the phone (and watch) buzz.

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
   the priority header alone overrides an iPhone Focus configuration.
5. The Apple Watch mirrors iPhone notifications automatically - no extra
   setup.

Alert semantics: each side (above/below) fires **once per arm**, then
disarms so a chopping price can't spam you; re-arm from the panel (or
**Re-arm All** for the next night). Levels are checked about once a minute,
pre/post market included (01:00-17:00 local). Every triggered crossing pushes
at urgent priority, lands in the Alert Center, and is listed in the morning
briefing. Monitoring runs in every mode by default - untick "Monitor in every
mode" to restrict it to EVENING. Only the main-desk alert engine monitors; the
designated shared-store writer check remains a second guard, so satellites and
the mini-PC never double-push one cross.

Protected-topic option: create/reserve the topic and an access token in ntfy,
enter that token in the TradingBot token field, and authenticate the phone's
subscription to the same protected topic. TradingBot sends the token as a
Bearer credential over HTTPS. Self-hosting is optional and requires the phone
to subscribe to that server too; simply changing the desktop URL is not enough.

## Morning arrival

- Slept to 07:30+? The briefing is final - read it in `evening_briefing.txt`,
  the phone report, or the Alert Center line, approve/pass the staged picks
  off their charts, then click the mode off EVENING.
- At the desk before 07:00? Nothing is lost: the checks and early scan run on
  schedule anyway; switch to DESK whenever you want live flow back.
