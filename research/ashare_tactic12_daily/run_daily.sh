#!/bin/zsh
# LaunchAgent 入口：每个工作日 09:00 运行；非交易日由 paper.py 跳过。
cd "/Volumes/Disk/项目/Vibe-Trading/research/ashare_tactic12_daily" || exit 1
export PYTHONPATH="/Volumes/Disk/项目/Vibe-Trading/research"
LOG=/Volumes/Disk/项目/data-cache/tactic12_paper/logs; mkdir -p $LOG
/Volumes/Disk/项目/Vibe-Trading/.venv-hermes/bin/python -u paper.py --send "$@" >> $LOG/run_$(date +%F).log 2>&1
