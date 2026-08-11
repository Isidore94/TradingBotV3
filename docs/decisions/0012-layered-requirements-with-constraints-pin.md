# 0012 — Layered requirements files pinned by constraints.txt

Date: backfilled 2026-08-01

Environment amendment: 2026-08-10 — the repo `.venv` is uv-managed Python 3.12 and
contains no pip. Use `uv pip ... --python .venv\Scripts\python.exe`.

## Context
Headless tools need a smaller dependency layer while the main desk and development
environment need the full GUI/test/packaging stack. Installs must be reproducible.

## Decision
Dependencies are layered: `requirements-core.txt` (headless engines/data) ⊂
`requirements-gui.txt` (adds Qt stack) ⊂ `requirements-dev.txt` (adds pytest,
pyinstaller); `requirements.txt` is a compatibility alias for the GUI layer.
Install with `uv pip install -r <layer> -c constraints.txt --python
.venv\Scripts\python.exe`; `constraints.txt` pins the known-good environment.

## Rationale
Documented in the `pyproject.toml` header comment (plan.md Phase 10.8/10.9): "one
place for test/lint configuration and a reproducible-install story."
