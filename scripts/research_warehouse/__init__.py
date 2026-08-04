"""Research warehouse: the DAS-hosted immutable Parquet research lake.

Locked plan: docs/ULTIMATE_SETUP_DATABASE_PLAN.md (Phases 0-8). This package
holds only additive, shadow-only evidence infrastructure - it never influences
a detector, score, ranking, or alert. The lake is a separate storage class
from the Drive home folder (docs/decisions/0014-das-research-lake.md); nothing
operational moves here, and the lake never lives inside the Drive folder.
"""
