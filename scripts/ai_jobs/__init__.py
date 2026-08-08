"""Local AI batch layer (plan.md item 13b, docs/LOCAL_AI_AUTOMATION_PLAN.md).

Off-hours jobs that read the desk's own evidence and write advisory documents
a human reads. Nothing in this package may write scanner state, scores,
watchlists, alerts, or order state, and no output feeds a detector, a score,
an alert decision, or a state machine.

The package is deliberately importable headlessly: it depends on
``requirements-core.txt`` only, so the standalone runner
(``scripts/run_ai_jobs.py``) never needs Qt and never needs the trading GUI to
be up. That separation is the point -- the desk is meant to be running during
market hours, and this layer is meant to be running when it is not.
"""

from __future__ import annotations
