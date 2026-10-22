# Data audit — QM-EOD-v2

- Panel: reused `qm_research_001` `panel.npz` (sha256 `cfa4573bd5a20f79…`)
- Source: `~/projects/trading/data/ts_live/` daily bars; cutoff `2026-10-20`
- EDGAR 8-K 2.02: reused; tags optional for EP
- Minute bars: **not available** → nature = **EOD-DIAG**
- Survivorship: pool is current-listing snapshot → `UNIVERSE_SURVIVORSHIP_BIAS` (optimistic)
- EP-C2 (PIT earnings surprise): `NOT_RUN_DATA_GAP`
- No short / no borrow path this round
