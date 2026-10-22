# result.md — QM-EOD-v2 给派活人

**run_id:** `qm-eod-v2-20261022-1056`  
**结果路径:** `/Users/jake/projects/trading/Vibe-Trading/research/qm_eod_v2/results/qm-eod-v2-20261022-1056`  
**结论（三选一）: 证据不足**

信号层期望为正（1.05R）但账户层弱（年化 0.83% vs QQQ 18.86%）；多持仓已测，仍缺分钟线与退市股。继续研究需补数据，不宜升级为可交易结论。

## 关键数字（账户层 / 信号层）

- 账户：CAGR 0.83%，MDD -16.11%，Sharpe 0.17，在场 47.87%，均持仓 1.03
- 对照 QQQ CAGR 18.86%
- 信号：n_signals=2111，fills=594，执行率 0.314，avg_R=1.05，PF=1.08
- 成本 25/50 bps 见 `metrics.csv` / `summary_zh.md`
- 性质：EOD-DIAG + UNIVERSE_SURVIVORSHIP_BIAS；未做空；EP-C2 NOT_RUN_DATA_GAP

详见 `summary_zh.md`、`concentration.md`、`acceptance.md`。
