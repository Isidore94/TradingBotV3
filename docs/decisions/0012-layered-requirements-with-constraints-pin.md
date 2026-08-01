# 0012 — Layered requirements files pinned by constraints.txt

Date: backfilled 2026-08-01

## Context
The mini-PC runs headless engines while the desktop needs the full GUI and dev
tooling; installs must be reproducible across both machines.

## Decision
Dependencies are layered: `requirements-core.txt` (headless engines/data) ⊂
`requirements-gui.txt` (adds Qt stack) ⊂ `requirements-dev.txt` (adds pytest,
pyinstaller); `requirements.txt` is a compatibility alias for the GUI layer.
Install with `pip install -r requirements.txt -c constraints.txt`, where
`constraints.txt` is regenerated from the known-good venv (`pip freeze`).

## Rationale
Documented in the `pyproject.toml` header comment (plan.md Phase 10.8/10.9): "one
place for test/lint configuration and a reproducible-install story."
