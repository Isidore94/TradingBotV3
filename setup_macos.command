#!/bin/bash
# One-time macOS setup: create the repo-local .venv and install the GUI stack.
# Re-running is safe; it reuses the venv and upgrades to the pinned versions.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${TRADINGBOTV3_PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "python3 was not found. Install Python 3.12+ from https://www.python.org or 'brew install python'." >&2
    exit 1
fi

"$PYTHON" - <<'EOF'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(
        f"Python 3.12+ is required; found {sys.version.split()[0]}. "
        "Point TRADINGBOTV3_PYTHON at a newer interpreter."
    )
EOF

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-gui.txt -c constraints.txt

echo
echo "Setup complete. Launch the desk with: .venv/bin/python launch_gui.py"
