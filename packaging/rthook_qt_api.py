"""Pin the Qt binding before anything imports qtpy.

QtAwesome and QtPy choose a binding at import time from whatever they can find.
A dev venv may still carry PyQt5 from before 2026-09-03 (it left the
dependency set with TickerMover.py), so without this the frozen app could
bind to the wrong one. The
spec excludes PyQt5 from the bundle; this makes the intent explicit rather than
relying on the exclusion alone, and gives a clear failure if it were ever
reintroduced.
"""

import os

os.environ.setdefault("QT_API", "pyside6")
