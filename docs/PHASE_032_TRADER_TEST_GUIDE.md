# Phase 0.32 trader test guide

This tests the new **Entry quality** and **Next test** work.

This work is research only. It must not change alerts, picks, scores, watchlists, or trades.

## Before you start

1. Start the Trading Desk the normal way.
2. Let it finish loading.
3. Keep it open through one normal session and the next overnight AI run if you can.

You do not need to type a command.

## Test 1 — The desk still works

1. Open the main Desk page.
2. Open a chart.
3. Open the Focus list and the setups list.
4. Make sure the desk does not freeze or close.
5. Make sure your normal alerts, picks, and watchlists still look normal.

Pass: the desk works like it did before.

## Test 2 — Look at the new Next test card

1. Open **Daily Recap**.
2. Open the **Review** tab.
3. Find the box named **Next test**.

You may see:

- A real test question. This means enough measured data was ready.
- `No validated proposal has been published yet`. This means it is still waiting for measured data or the overnight AI run. This is not a failure.

If a test question is shown, the card should also show:

- the control;
- one thing to change;
- eligible, no-trigger, and missing counts;
- limits and status;
- a report number and hash;
- when the proposal and its source were made.

Pass: the card shows facts, or clearly says it is still waiting. It must never pretend missing data is zero.

## Test 3 — Copy the test brief

Do this only when a real Next test question is shown.

1. Click **Copy test brief**.
2. Paste it into Notepad.
3. Compare it with the card.

Pass:

- the question matches;
- the report number and hash match;
- the control and one change match;
- clicking Copy does not start a test, change a pick, or create an alert.

## Test 4 — Check the Setup Tracker link

1. Open **Research**.
2. Open **Setup Tracker**.
3. Find its **Next test** line.
4. If a real question is shown, click **Open Next test in Review**.

Pass: it opens Daily Recap's Review tab and shows the same question.

If both places say no proposal exists yet, that is still a pass while data is collecting.

## Test 5 — Check it again after the overnight run

1. Leave the desk open for the normal overnight AI run.
2. The next day, open **Daily Recap > Review** again.
3. Check the Next test card and Copy test brief again.

Pass:

- the screen and copied brief agree;
- old measured facts do not silently change;
- new progress may change the counts;
- missing evidence says `not measured`, `unknown`, or `collecting`;
- no screen calls a setup proven or best from this new work.

## Stop and tell Codex if

- the desk freezes or closes;
- Daily Recap and Setup Tracker show different questions;
- the card and copied brief show different report numbers or hashes;
- Copy test brief starts anything;
- a Next test changes an alert, score, pick, Focus name, or watchlist;
- missing data is shown as zero;
- the card claims a winner or proven setup.

## What Codex still has to test

Codex still needs to run four checks on safe copies of the data:

1. Fixed-time entry movement is measured correctly.
2. Entry choices are compared fairly.
3. The report, card, Tracker, and copied brief all use the same facts.
4. The local AI uses only those facts, and its speed and memory are measured.

Those are gates **#140–#143**. They do not require you to change live data.

## What to send back

Send Codex:

- a screenshot of the Next test card;
- the text from **Copy test brief**;
- the time you checked it;
- one short sentence about anything that looked wrong.
