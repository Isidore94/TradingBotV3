# Alert Center quality packet

Status: **Phases 0–2 landed and green, Phase 3 partly.** The Section 9
ask-first gate is answered. Phase 4 is the live exit gate and needs real
sessions on the desk — the code is GREEN, not `LIVE_VALIDATED`.

Branch: `claude/alert-center-quality-packet-5btu3w`.

Subordinate to `plan.md`. This packet does not override Section 5 invariants,
the Section 7 promotion ladder, or the Section 12 order. It implements the
measurement surface named in `GUI_TRADE_DISCOVERY_LEARNING_PLAN.md` sec 10.3
("Alert Quality" dashboard) and sec 17 ("Alert quality" metrics), which are
currently specified but unbuilt.

## 1. The problem in one paragraph

The desk decides, every alert, whether to make noise — `alert_is_loud()` and
`alert_should_sound()` (`scripts/ui/panels/alert_center_panel.py:211,229`) gate
a `QApplication.beep()` at line 832. That decision is never recorded. The
review log (`review_events.py`, `review_events_v2`) captures what the trader
*did* — `shown`, `skip`, `add_focus`, `arm_watch`, `watch_fired` — but never
what the desk *delivered*. The result: nobody can answer "how many loud alerts
did I get today, and how many were the same name shouting twice?" There is no
duplicate-rate number, no loud-per-session number, and no delivery latency for
a watch the trader explicitly armed. The learning loop measures the trader; it
does not measure the alerting.

## 2. Scope

Measurement and evidence only. This packet adds capture and a scoreboard. It
changes no detector, no score, no ranking, no alert gate, and no sound
decision. `alert_is_loud()` and `alert_should_sound()` are read, never edited.

Because no champion behavior changes, no golden fixtures are required — the
same basis on which plan.md items 13a and 13b proceed. The Section 9 gate below
still applies, because the capture site lives inside an alert file.

### Explicitly out of scope

- Any change to what is loud, what sounds, or what is queued.
- Any suppression path. `review_policy.json` deliberately has no suppression
  field (CLAUDE.md invariant; `docs/REVIEW_LEARNING_LOOP.md`) and this packet
  does not add one, anywhere. Muted means CAUTION, not silence.
- Any change to queue ordering. The FIFO annotation-only gate
  (`review_guidance.py`, `ORDERING_ANNOTATION_ONLY`) stays exactly as it is;
  measuring duplicates is not a licence to reorder on them.
- Deduplicating alerts. This packet learns the duplicate rate. Acting on it is
  a separate, later, trader-approved decision.

## 3. What is measurable today, and what is not

`GUI_TRADE_DISCOVERY_LEARNING_PLAN.md` sec 17 defines seven alert-quality
metrics. Audited against current capture:

| Metric (sec 17 / sec 10.3) | Before this packet | After | Note |
|---|---|---|---|
| Alert-to-action conversion by action type | Partly | Partly | Covers reviewed alerts only; an alert that never reached the review queue leaves no impression to divide by |
| Watch conversion | Yes | Yes | Already in `review_learning.py` |
| Loud alerts per session | No | **Yes** | Unblocked by Phase 1 delivery capture |
| Duplicate loud rate | No | **Yes** | Needed a typed `alert_event_id`, which did not exist in the repo; Phase 1 adds it |
| User-armed hit delivery rate + latency | No | **Yes** | `watch_fired` records the fire; `watch_delivered` adds the visible delivery and its latency |
| Missed-winner rate among quiet/queued items | No | No | The quiet cohort is still not logged — delivery capture records what WAS shown, not what was withheld |
| Ready precision / precision@K | No | No | Gated on canonical Ready — **deferred, not in this packet** |
| Remaining Expected R at alert | No | No | Needs versioned target/stop at alert time; deferred with Ready above |

Two conclusions drive the phase order. First, four of the unbuilt metrics share
exactly one blocker — there is no delivery record — so one capture change
unlocks all four. Second, Ready precision and Remaining Expected R depend on
the canonical Ready lifecycle, which this packet does not build and must not
pretend to; they are named here and deferred, not silently dropped.

## 4. The identity problem (`alert_event_id`)

The plan's duplicate definition is "loud deliveries after the first for the
same typed `alert_event_id` without a genuine escalation". That identifier does
not exist. `BounceAlert` carries an `event_id` joining to
`intraday_bounce_candidates.csv`, but it is per-candidate-row, not per typed
alert occurrence, and D1/chart-watch/level alerts do not participate.

Proposed identity, to be confirmed:

```
alert_event_id = (trade_date, symbol, side, alert_type, thesis_anchor)
```

where `alert_type` is the typed family (m5_bounce, d1_event, d1_level,
chart_watch, focus_pick, status) and `thesis_anchor` is the family's natural
anchor — the watch kind for a chart watch, the level+direction for a level
alert, the bucket/setup family for D1. Deliberately *not* keyed on bar
timestamp, or every re-fire becomes a distinct alert and the duplicate rate is
identically zero.

"Genuine escalation" needs a trader-facing definition before the metric means
anything. Proposed: a delivery is an escalation, not a duplicate, when the tier
rises, a quiet alert becomes loud, or an armed condition fires on a name that
was previously only queued. This is a judgement call and is listed in Section 9
as a question, not an assumption.

## 5. Phases

### Phase 0 — capture-gap audit and definition freeze (no alert-file edits)

New read-only module `scripts/alert_quality.py` plus
`tests/test_alert_quality.py`. Reads the existing merged review-event shards and
reports which sec 17 metrics are computable, with sample counts and date range,
and prints honest `Unknown` for the rest rather than a zero. Establishes the
metric registry — one `outcome_definition_id`, horizon, and cohort rule per
metric, per the sec 17 preamble — so later numbers are comparable.

Touches no alert file. Safe to build before the Section 9 answer, and is the
only phase to start unprompted.

**Landed** (`scripts/alert_quality.py`, `tests/test_alert_quality.py`, 25
tests): the metric registry with a frozen `outcome_definition_id` per metric,
the capture-coverage audit (rows, sessions, span, writers, schemas, and a
multi-installation warning), the delivery-gap finding, and the two metrics that
are computable today. An empty store renders as an empty store, explicitly not
as a quiet desk.

### Phase 1 — delivery capture (touches alert files) — **LANDED**

Two actions, written at the existing `add_alert` and watch-poll sites:

- `delivered` — one row per alert reaching the feed, carrying
  `alert_event_id`, `alert_type`, `loud`, `sounded` (whether a beep actually
  happened), `is_focus`, `tier`, and the existing structured context.
- `watch_delivered` — the visible delivery of a fired armed condition,
  carrying `watch_id` and `fired_to_delivered_ms`, which is what makes the
  sec 17 latency bound measurable.

Capture-side only. The beep, the gates, and the queue are untouched; the panel
diff is +87/−3 and contains no gate or threshold. `alert_is_loud` is read
rather than re-implemented, so the recorded judgement cannot drift from the one
driving the beep. Best-effort and exception-swallowing like every other emit
site, and tests assert that a raising recorder still leaves the alert in the
feed and does not stop the next one.

**Design changed during implementation** (from `review_events_v3` to a separate
store), because the trader's storage decision made it the right shape:

- Delivery rows go to a new machine-local store
  (`scripts/alert_delivery_events.py`, schema `alert_delivery_events_v1`) under
  the diagnostics root, partitioned by month — **not** the Drive-synced review
  store. Keeping them out of `review_events` means the synced schema never
  moves, so there is no reader/writer skew risk on the cloud-synced file at all,
  which the v3 bump would have introduced for no benefit.
- Escalation is **not** decided at write time. Rows carry the inputs (`tier`,
  `loud`, `is_armed_fire`) and the reader applies the rule, so revising the
  definition costs a re-read rather than re-instrumenting and waiting another
  month for data.
- There are **three** armed-condition types, not one — chart watches, D1 price
  levels, and D1 event watches. All three record deliveries and all three count
  in the armed-hit denominator; wiring only the first would have silently
  excluded every level and D1 event the trader armed. A price level is
  identified by direction and price rather than a `kind`, and both stores build
  that identity component identically or the join yields an empty intersection
  that looks like a delivery failure.
- `watch_id` is **derived** from `(trade_date, symbol, side, kind)` on both
  sides rather than added to the existing `watch_fired` payload — the smallest
  change that makes the join possible without editing another emit site inside
  the alert panel.
- The measured latency is **detection → on screen**. Detection lag relative to
  the bar close is a separate quantity this does not measure, and the metric
  must not be read as covering it.

### Phase 2 — the scoreboard — **LANDED**

`alert_quality.py` computes loud per session, duplicate loud rate with the
escalation carve-out, and armed-hit delivery rate with latency. Every output
carries independent sample count, session count, and date range per the
sec 10.4 evidence display rules; below the sample floor it prints the floor and
`Unknown`.

Three places the easy implementation would have flattered the desk, and does
not: a fired watch never delivered stays in the denominator instead of
vanishing from both sides; a delivery with no recorded latency cannot prove it
met the bound, so it is not a hit; and the per-session floor counts sessions
rather than alerts, so one very loud day is not evidence about normal volume.

### Phase 3 — surface — **partly landed**

Landed: the `review_delivery_capture` check in `review_capture_audit.py`, which
states whether delivery capture is actually running, so a silently-dead emit
site is visible rather than mistaken for a quiet session. It renders in System
Health through the existing audit plumbing with no new panel code, and is
capped at `degraded` — advisory evidence with no path to a detector cannot take
the desk red.

Remaining: a dedicated Alert Quality panel section rendering the Phase 2
numbers. The CLI (`scripts/alert_quality.py`) is the interim surface, and is
the right one until the exit gate proves the numbers are worth a panel.

### Phase 4 — exit gate

The packet is not done when it is green. It is done when the numbers are
trustworthy:

- ≥10 sessions of delivery capture on the real desk.
- Delivered-row count reconciles with feed-item count for those sessions;
  a systematic gap means the emit site is wrong.
- The duplicate rate is hand-checked against one full session the trader
  actually remembers, and it matches their experience. If the number says the
  desk is quiet on a day they remember as noisy, the metric is wrong, not the
  trader.
- Full suite green + smoke 7/7 before each commit.

## 6. Invariants this packet must not break

- Decision-support only; no execution path is touched.
- Legacy SPY pause detection and D1 wick alerts stay champion. Nothing here
  influences them.
- Completed bars only for state transitions. This packet reads no bars and
  makes no transitions.
- No suppression field, ever.
- Ordering stays FIFO under the annotation-only gate.
- One component owns each mutable export; `alert_quality.py` is a reader and
  owns nothing.
- A failed publish never destroys the last verified report.

## 7. Test plan — landed

- `tests/test_alert_quality.py` (45) — metric maths on synthetic logs: the
  empty log, the below-floor log, each arm of the escalation rule, and the
  cases where the easy implementation would flatter the desk.
- `tests/test_alert_delivery_events.py` (23) — storage class, typed identity,
  and that a malformed alert cannot break a write. The isolation fixture is
  `autouse`: an opt-in one let an early version append synthetic rows to the
  real diagnostics directory, which conftest.py exists to forbid.
- `tests/test_qt_alert_delivery_capture.py` (7) — offscreen Qt: one row per
  delivered alert, nothing for a suppressed one, recorded loudness matching the
  panel's own verdict, and a raising recorder neither losing an alert nor
  stopping the next.
- `tests/test_review_capture_audit.py` (+4) — the health check tells a stopped
  emit site apart from a quiet desk, and never goes past `degraded`.

## 8. Risks

- **Log volume.** A delivery row per alert is far higher-volume than a decision
  row. Mitigated by the machine-local, month-partitioned store — retention is
  deleting old files, and nothing syncs — but the real per-session row count is
  still unmeasured and is an exit-gate item.
- **Escalation definition is a judgement call.** Get it wrong and the duplicate
  rate is either ~0 or ~1 and useless either way. Mitigated by applying the
  rule at read time: revising it costs a re-read, not another month of capture.
- **Measurement inviting action.** The first instinct on seeing a duplicate
  rate will be to suppress duplicates. That is a separate decision through the
  plan's Section 7 ladder, not a follow-on commit.
- **Latency semantics.** `fired_to_delivered_ms` covers detection → on screen.
  If it is ever read as fire → on screen it will understate the trader's real
  wait, because detection lag is not in it.

## 9. Ask-first gate — answered 2026-08-18

The file-scoped rule (CLAUDE.md, checkpoint review 2026-08-08) required the
trader's answer before Phase 1 touched `alert_center_panel.py`. Answers:

1. **Reading of "quality":** *alert quality* — measuring what the desk
   delivers. Confirmed.
2. **Escalation:** tier rises, quiet becomes loud, or an armed condition fires.
   Confirmed as written in Section 4.
3. **Volume:** machine-local shards only, never the Drive-synced store.
   Confirmed, and it is what changed Phase 1's design.
4. **Identity / Ready metrics:** unchanged from Section 4 and Section 3 — the
   typed tuple stands, and Ready precision / Remaining Expected R stay deferred
   to the canonical-Ready work rather than being approximated here.

## 10. Status

- **Phases 0–2: landed.** Capture-gap audit, machine-local delivery store, and
  the scoreboard.
- **Phase 3: partly landed.** The health check is in; the dedicated panel
  section is not. `.venv\Scripts\python.exe scripts/alert_quality.py [--days N]`
  is the interim surface.
- **Phase 4 (exit gate): not started, and cannot be from here.** It needs ≥10
  real sessions of capture on the desk, a delivered-row/feed-item
  reconciliation, and the duplicate rate hand-checked against a session the
  trader remembers. Until then the code is **GREEN**, not `LIVE_VALIDATED`.

Suite green: **2131 passed, 5 skipped, 7 subtests** (79 new), smoke **7/7**,
ruff clean on every touched file.

### First run on the desk

The first `scripts/alert_quality.py` run after this lands will report `Unknown`
for the delivery-backed metrics and `degraded` for `review_delivery_capture`.
That is correct: capture starts writing only once the GUI restarts onto a build
containing it. Numbers appear from the first session onward, and the sec 17
rates hold at `Unknown` until they clear the sample floor.

### Note on the test baseline

The suite is fully green only under a Pacific timezone
(`TZ=America/Vancouver`). Several existing tests — `test_vold_recorder.py`,
`test_breadth_backfill.py`, `test_autopilot_core.py` and others — build
`America/Vancouver` timestamps and parse IB time strings in local time, so a
UTC machine fails 16 of them for reasons that have nothing to do with the code
under test. This is a pre-existing property of the suite, not something this
packet introduced or fixed; it is recorded here so the next agent on a
non-Pacific box does not mistake it for a regression.
