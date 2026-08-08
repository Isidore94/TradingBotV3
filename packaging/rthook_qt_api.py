"""Pin the Qt binding before anything imports qtpy.

QtAwesome and QtPy choose a binding at import time from whatever they can find.
The dev venv has PyQt5 installed too (it survives only for the legacy
TickerMover.py), so without this the frozen app can bind to the wrong one. The
spec excludes PyQt5 from the bundle; this makes the intent explicit rather than
relying on the exclusion alone, and gives a clear failure if it were ever
reintroduced.
"""

import os

os.environ.setdefault("QT_API", "pyside6")
