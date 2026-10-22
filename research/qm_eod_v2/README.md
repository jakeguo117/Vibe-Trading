# QM-EOD-v2

Frozen EOD diagnostic (BO+EP, max 5 positions) per `spec/QM_EOD_v2_Spec_20261022.md`.

```bash
export QM_EOD_RUN_ID=qm-eod-v2-YYYYMMDD-HHMM
# symlink panel.npz + edgar into results/$QM_EOD_RUN_ID/data/ from qm_research_001
nice -n 15 ../../.venv/bin/python scripts/run_all.py
../../.venv/bin/python scripts/report.py
```

Nature: **EOD-DIAG** — not intraday ORH.
