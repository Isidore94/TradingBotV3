# Alert Center quality packet

Status: **Phase 0 landed and green.** Phases 1-4 are blocked on the trader's
answer to the ask-first gate in Section 9.

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

| Metric (sec 17 / sec 10.3) | Computable today? | Blocker |
|---|---|---|
| Alert-to-action conversion by action type | **Partly** | `shown` + action rows exist; folds episodes by `(trade_date, symbol)`, so a long and a short collapse |
| Watch conversion | **Yes** | already in `review_learning.py` |
| Loud alerts per session | **No** | no delivery row is ever written |
| Duplicate loud rate | **No** | needs a typed `alert_event_id`; the identifier does not exist in the repo |
| User-armed hit delivery rate + latency | **No** | `watch_fired` records the fire, not the visible delivery or its timestamp |
| Missed-winner rate among quiet/queued items | **No** | requires the quiet cohort, which is never logged |
| Ready precision / precision@K | **No** | gated on canonical Ready, which is Phase 3+ of the GUI learning plan — **not this packet** |
| Remaining Expected R at alert | **No** | needs versioned target/stop at alert time; gated with Ready above |

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

### Phase 1 — delivery capture (**gated, touches alert files**)

Two new review-event actions written at the existing `add_alert` site:

- `delivered` — one row per alert reaching the feed, carrying
  `alert_event_id`, `alert_type`, `loud` (the recorded result of
  `alert_should_sound`, never a re-derivation), `sounded` (whether the sound
  checkbox was actually on), and the existing structured context.
- `watch_delivered` — the visible delivery of a fired watch, carrying
  `watch_id` and `fired_to_delivered_ms`, which is what makes the sec 17
  latency bound measurable.

Capture-side only: `add_alert` gains a `record_review_event(...)` call and
nothing else. The beep, the gates, and the queue are untouched. Best-effort and
exception-swallowing like every other emit site — a cloud-synced folder locking
the log must never surface as a GUI error, and must never drop an alert.

Schema goes to `review_events_v3`, additive. Readers already merge legacy
shards; `SUPPORTED_REVIEW_EVENTS_SCHEMAS` grows, and no reader may require the
new fields, so old shards stay readable and the desk can roll back to a v2
build without losing the log.

### Phase 2 — the scoreboard

`alert_quality.py` grows the real computations over Phase 1 data: loud per
session, duplicate loud rate with the escalation carve-out, armed-hit delivery
rate and latency distribution, and alert-to-action conversion split by
`alert_type` so a long and a short no longer collapse. Every output carries
independent sample count, session count, date range, and dispersion, per the
sec 10.4 evidence display rules. Below the sample floor it prints the floor and
`Unknown`.

### Phase 3 — surface

An Alert Quality section in System Health, rendered from Phase 2 like the
existing audits (`operations_audit.py`, `review_capture_audit.py`). Read-only.
Plus a `review_capture_audit.py` line stating whether delivery capture is
actually running, so a silently-dead emit site is visible rather than being
mistaken for a quiet session.

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

## 7. Test plan

- `tests/test_alert_quality.py` — metric maths on synthetic event logs,
  including the empty log, the below-floor log, and the escalation carve-out.
- Schema round-trip: a v2 shard and a v3 shard merged by one reader.
- Offscreen Qt test asserting `add_alert` emits exactly one `delivered` row per
  alert, and that a raising `record_review_event` cannot lose an alert or
  surface an error.
- A characterization test pinning `alert_is_loud` / `alert_should_sound` before
  Phase 1, so the packet can prove it changed no gate.

## 8. Risks

- **Log volume.** A delivery row per alert is far higher-volume than a decision
  row. Needs a measured estimate against a real session before Phase 1 merges;
  the shard directory is on the cloud-synced home folder.
- **Escalation definition is a judgement call.** Get it wrong and the duplicate
  rate is either ~0 or ~1 and useless either way.
- **Measurement inviting action.** The first instinct on seeing a duplicate
  rate will be to suppress duplicates. That is a separate decision through the
  Section 7 ladder, not a follow-on commit.

## 9. Ask-first gate — open questions

The file-scoped rule (CLAUDE.md, checkpoint review 2026-08-08) requires the
trader's answer before Phase 1 touches `alert_center_panel.py`, even for
capture-only changes.

1. **Is this the right reading of "quality"?** This packet reads it as *alert
   quality* — measuring what the desk delivers. It could instead have meant
   code quality of a 2,776-line panel, or review-capture completeness. The
   answer changes the whole packet.
2. **Escalation:** is the Section 4 definition right?
3. **Identity:** is the Section 4 `alert_event_id` tuple the right grain?
4. **Volume:** is a delivery row per alert acceptable in the Drive-synced log,
   or should delivery capture shard machine-locally only?
5. **Ready metrics:** confirm Ready precision / Remaining Expected R stay
   deferred to the canonical-Ready work rather than being approximated here.

## 10. Status

- **Phase 0: landed.** Suite green (2077 passed, 5 skipped, 7 subtests; 25
  new), smoke 7/7. Run it with
  `.venv\Scripts\python.exe scripts/alert_quality.py [--days N]`.
  Against a real store it will currently report `Unknown` for four of the
  seven sec 17 metrics — that output *is* the Phase 0 finding, not a defect.
- **Phases 1-4: blocked** on the Section 9 answers. Phase 1 edits
  `alert_center_panel.py`, so it does not start without them.

### Note on the test baseline

The suite is fully green only under a Pacific timezone
(`TZ=America/Vancouver`). Several existing tests — `test_vold_recorder.py`,
`test_breadth_backfill.py`, `test_autopilot_core.py` and others — build
`America/Vancouver` timestamps and parse IB time strings in local time, so a
UTC machine fails 16 of them for reasons that have nothing to do with the code
under test. This is a pre-existing property of the suite, not something this
packet introduced or fixed; it is recorded here so the next agent on a
non-Pacific box does not mistake it for a regression.
