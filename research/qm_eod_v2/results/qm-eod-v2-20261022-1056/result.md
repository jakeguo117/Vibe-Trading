# result.md — QM-EOD-v2 给派活人

**run_id:** `qm-eod-v2-20261022-1056`  
**绝对路径:** `/Users/jake/projects/trading/Vibe-Trading/research/qm_eod_v2/results/qm-eod-v2-20261022-1056`  
**commit:** `2dba0175` on `research/qm-eod-v2` (pushed)  
**结论（三选一）: 证据不足**

信号层期望为正（avg_R=1.05R，PF=1.08，n=594）但账户年化仅 0.83%、远低于 QQQ 18.86%；前5笔占净利 165%（集中度极高）。多持仓（max5）已测、执行率升至 31.4%，仍属 **EOD-DIAG**，缺分钟线与退市股，不能写成原版 ORH 通过/证伪。

## 账户层（FULL，filter ON，10bps，$100k）

| 指标 | 策略 | QQQ |
|---|---:|---:|
| CAGR | 0.83% | 18.86% |
| MDD | -16.11% | -33.95% |
| Sharpe | 0.17 | 0.92 |
| 在场 | 47.9% | 100% |
| 均持仓 | 1.03（峰值5） | — |
| 终值 | 109,521 | — |

成本：10bps CAGR 0.83% → 25bps −0.43% → 50bps −2.65%。无过滤对照 CAGR 1.85%。$1k 对照因最低佣金相对名义过大而严重侵蚀。

## 信号层

| 项 | 值 |
|---|---:|
| 信号 | 2111（BO 1194 / EP 917） |
| 成交 | 594（BO 393 / EP 201） |
| 执行率 | 31.4% |
| avg_R / med_R | 1.05 / -1.02 |
| PF / 胜率 | 1.08 / 34.8% |

拒单主因：qqq_filter 804、stop_too_wide 300、max_positions 188。

## 规格偏离（已写入 run_manifest）

1. BO 横盘长度用离散 {10,15,20,25,30,35,40} 扫描，非逐 bar 连续 10–40 搜索（速度）。
2. EP 主结果用价格跳空；8-K 仅打标（符合规格「不强制」）。
3. 主路径为 BO+EP 合并 5 槽账户；BO/EP 单独为诊断。

性质：EOD-DIAG + UNIVERSE_SURVIVORSHIP_BIAS；未做空；EP-C2 NOT_RUN_DATA_GAP。
