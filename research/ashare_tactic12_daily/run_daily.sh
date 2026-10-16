#!/usr/bin/env bash
# 通用入口（Mac LaunchAgent / GitHub Actions 均可）。所有路径由环境变量决定：
#   STATE_DIR  纸面账户状态、订单表、发送标记（默认 Mac 路径）
#   CACHE_DIR  日线缓存（bars/*.parquet）
#   PYTHON     解释器（默认 python3）；LOG_DIR 设置时把输出追加到日志文件
# 额外参数透传给 paper.py，例如 --dry-run / --today 2026-10-19
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PYTHONPATH="$(dirname "$HERE")${PYTHONPATH:+:$PYTHONPATH}"
export STATE_DIR="${STATE_DIR:-/Volumes/Disk/项目/data-cache/tactic12_paper}"
export CACHE_DIR="${CACHE_DIR:-/Volumes/Disk/项目/data-cache/ashare_tactics}"
PY="${PYTHON:-python3}"
if [[ -n "${LOG_DIR:-}" ]]; then
  mkdir -p "$LOG_DIR"
  exec "$PY" -u paper.py --send "$@" >> "$LOG_DIR/run_$(date +%F).log" 2>&1
fi
exec "$PY" -u paper.py --send "$@"
