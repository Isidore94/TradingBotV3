# Wishlist items that need one trader decision before they can be built

Status: **active** — one question per item, each blocking a build that is
otherwise ready. Created 2026-08-18 under the trader's integration redirect,
which asked for every implementable `WISHLIST.md` item to be built and for
anything "too vague to build without a trading judgment" to get a spec stub
stating the open question instead of a guess.

Read this beside `WISHLIST.md`, which stays the trader-visible idea list. This
file is not a roadmap and nothing here is authorized: an item leaves this page
by the trader answering its question, at which point it follows the normal
promotion path into `plan.md`.

**Why a stub rather than a best guess.** Each item below has exactly one
unanswered question whose plausible answers lead to *different code*, not
different polish — a different storage location, a different data budget, a
different failure posture. Building the wrong branch and calling it a default
would hide the decision inside an implementation, which is the thing the
ask-first rule exists to prevent.

---

## Built instead of stubbed (2026-08-18)

| Item | What landed |
|---|---|
| Deep-link a symbol/timeframe into TradingView or TC2000 | `scripts/external_chart_links.py` plus an **Open in TradingView** button on the arm bar, so every chart surface that carries the bar inherits it. The URL is a machine-local setting (`external_chart_url_template`), the symbol is validated before a URL is built, and a refused open is reported rather than swallowed. **TC2000 is deliberately not wired**: it is a desktop app whose documented automation surface is its own scripting layer, not a URL scheme, and a `tc2000://` link that silently does nothing would be worse than the honest gap — the template setting is the seam for it the day the trader confirms what their install answers to. |

---

## User-experience items

### Voice dictation for the live commentary journal

**Open question — local or cloud speech, and what happens to a bad
transcription?** Local (whisper.cpp on the 8845HS) keeps every word on the desk
and costs GPU time during a session; cloud is more accurate and sends the
trader's live commentary to a third party. The correction workflow follows from
that choice: a local model needs an edit-before-commit step, a cloud one could
commit and correct after.

Blocked also by ordering: `plan.md` P3.5 owns the commentary journal itself,
and there is nothing to dictate into yet.

### User-selectable chart line-density presets

**Open question — what does "too many lines" mean on the trader's screen?**
The prerequisite recorded in `WISHLIST.md` is P1.2's red-level threshold and
clutter budget, and that is a desk-evidence decision, not a preference toggle:
presets built before it would encode a guess about which levels matter, and the
trader would then be choosing between three wrong densities.

### Read-only mobile/web dashboard beyond the text digest

**Open question — who may read it, and from where?** A phone-reachable page
showing positions and candidates is an authentication and hosting decision
before it is a UI one, and the answer changes the whole build (a LAN-only page
behind the router is a different system from an internet-reachable one with
accounts). `plan.md` P5.3's one-snapshot work is the natural prerequisite.

### Self-hosted ntfy deployment

**Open question — is the operational burden worth it?** Hosted ntfy already
works and costs nothing to run. Self-hosting buys privacy and control and costs
TLS certificates, a reachable endpoint, backups, and a new way for the phone to
go silent on a Sunday. This is an operations judgment, not a code one; the
sender is already a thin seam, so the build is small once it is wanted.

### macOS equivalents for the Windows scheduled jobs

**Open question — will a Mac ever be the unattended host?** The wishlist entry
answers this conditionally ("only if macOS becomes an unattended host"), and
today the 8845HS is the sole always-on machine and sole writer. Building launchd
equivalents now would create a second scheduling surface that nothing runs and
nobody tests.

### Broader US universe for the M5 strength board

**Explicitly gated, not vague.** The trader chose the existing ~1,500-name
universe on 2026-08-15, and the entry says to widen it only after the R2 board
proves itself and a data/pacing budget is agreed. That is a live gate; it is
listed here only so the reason it was skipped is recorded.

---

## Research and data items

Every entry under "Candidate research and data integrations" shares one shape:
each needs a **registered consumer** before it may be captured. That is the
locked warehouse plan's own rule (`docs/ULTIMATE_SETUP_DATABASE_PLAN.md`) —
capture is justified by a question someone will actually ask, because an
append-only lake makes "collect it in case" a permanent cost.

| Item | The one question |
|---|---|
| DYNAMIC and EOD session-VWAP variants in the warehouse | Which registered study needs the variants? Capturing all three triples the VWAP surface and STANDARD must stay untouched either way. |
| Tier-2 anchor families (catalyst/gap bars, confirmed pivots, period opens) | Same: which study, and is the anchor's point-in-time availability provable at the simulated decision time? |
| Optional completed-bar tracker preview lane | The trader must ask for it. It is a labeled preview lane that can never write confirmed evidence, so its value is entirely about whether the trader wants an earlier, weaker read after a missed final scan. |
| Additional external universe feeds | What is the incremental value over the current sources (P5.5 measures it), and what is the pacing budget? Manual names can never be auto-removed regardless. |
| Options-chain / theta history | Data rights, IB pacing budget, and whether point-in-time quotes are actually retrievable. Any one of those can make the capture impossible rather than merely expensive. |
| News/catalyst archive with point-in-time availability | Which licensed source, and can published-vs-observed timestamps both be preserved? Without both, hindsight leaks into every study that reads it. |

---

## Not stubbed, deliberately

`TRIGGERED_LATER` items already carry explicit revisit triggers and
`PERMANENT_NO` items are settled product boundaries. Neither belongs here:
the first are waiting on a measurable event, and the second are not questions.
