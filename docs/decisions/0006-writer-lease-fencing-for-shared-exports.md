# 0006 — Writer leases with fencing generations for shared exports

Date: backfilled 2026-08-01

## Context
Both machines mount the shared Drive folder and can write the same export. A
Drive-synced file has no atomic test-and-set across machines, so true distributed
locking is impossible on this storage (decision 0005).

## Decision
`scripts/writer_lease.py` maintains a lease file next to each shared export naming
the writer, its process instance, and a monotonic fencing generation (with a durable
high-water mark). Ownership is re-checked before every shared replacement; every
ambiguous state fails closed. `scripts/local_writer_lock.py` and `writer_role.py`
handle same-machine and role coordination.

## Rationale
Fully documented in the `writer_lease.py` module docstring: it is deliberately
cross-machine writer *protection*, not mutual exclusion — ambiguous states must
"fail closed instead of fail open", and known residual races are stated openly.
