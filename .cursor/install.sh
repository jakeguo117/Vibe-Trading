#!/usr/bin/env bash
# Cursor Cloud Agent install (runs during each Build; must be idempotent).
# Vibe-Trading: editable install from pyproject.toml (NOT full requirements-lock.txt /
# docker build — those are too heavy for Cloud Agent Build time budgets ~5–10 min).
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), sys.version' || {
  echo "Python >= 3.11 required"; exit 1; }
[ -x .venv/bin/python ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install -q --upgrade pip
# Prefer project-standard editable install from pyproject (base deps only; no [dev]/extras).
# Do not pip install -r requirements-lock.txt here (lock ~370KB / full pin set is too slow).
# Do not docker build in this script.
python -m pip install -q -e .
# Top packages live under agent/ via setuptools package-dir; core package is `src`.
python -c 'import src'
echo "cursor install ok: $(python --version)"
