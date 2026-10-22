# Reproduce

```bash
cd ~/projects/trading/Vibe-Trading
git checkout research/qm-eod-v2
export QM_EOD_RUN_ID=qm-eod-v2-20261022-1056
nice -n 15 .venv/bin/python research/qm_eod_v2/scripts/run_all.py
.venv/bin/python research/qm_eod_v2/scripts/report.py
```

- SMA only (not EMA), as frozen.
- Panel/edgar symlinked from qm_research_001 run `qm-001-20261022-0041`.
- Combined BO+EP shared 5-slot book is primary.
