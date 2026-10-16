#!/usr/bin/env bash
# Cursor Cloud Agent install step (runs during each Build; must be idempotent).
# Creates .venv at the repo root with the project + test + A-share research deps.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), sys.version' || {
  echo "Python >= 3.11 required"; exit 1; }
[ -x .venv/bin/python ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install -q --upgrade pip
# vibe-trading-ai (this repo) in editable mode, same extras CI uses minus heavy/broker ones
python -m pip install -q -e ".[dev,stats]"
# research/ (ashare_tactics, ashare_tactic12_daily): data + parquet + JS decoder for Sina
python -m pip install -q akshare pandas numpy pyarrow requests py_mini_racer pytest
# smoke (offline): key imports; research tests when that folder exists on this branch
python -c "import akshare, pandas, pyarrow, pytest, agent" 
if [ -f research/ashare_tactics/test_logic.py ]; then
  ( cd research && python -m pytest -q ashare_tactics/test_logic.py $( [ -d ashare_tactic12_daily ] && echo ashare_tactic12_daily ) )
fi
echo "cursor install ok: $(python --version)"
