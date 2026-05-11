"""Compatibility imports for the Market Prep GUI package."""

from __future__ import annotations

import sys

from market_prep_gui import tabs as _tabs

sys.modules[__name__] = _tabs
