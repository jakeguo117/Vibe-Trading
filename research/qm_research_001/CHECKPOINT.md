# QM-RESEARCH-001｜断点与恢复（CHECKPOINT）

更新时间：2026-10-22 01:40 HKT｜run_id：`qm-research-001-20261022`

## 1. 已完成

| 阶段 | 状态 | 证据 |
|---|---|---|
| 读取方案 + 研究入口 | 完成 | `spec/`（sha256 见 config.json） |
| 数据审计（5 类） | 完成 | `data_readiness.md`、`runs/.../data/{coverage.csv,audit_summary.json,sample_verification.json}` |
| 股票池/日线面板 | 完成 | `panel.npz`（4,477 日 × 10,967 证券，2009–2026-10-20） |
| 财报事件（EDGAR 8-K 2.02） | 完成 | `edgar_8k202.csv`（74,538 条，1,518 证券） |
| 引擎自检 | 完成 13/13 | `test_evidence.md` |
| 冻结配置与评估清单 | 完成 | `config/{config,manifest}.json`（211 条评估） |
| 正式计算 | 完成 350/350 路径 | `checkpoint.json`、`deliverables/` |
| 验收/统计/归因/漏斗 | 完成 | `deliverables/{acceptance,statistics,attribution,funnel,comparison_table}.csv` |
| 报告 | 完成 | `00_summary_zh.md`、`statistics.md`、`attribution.md` |
| 代码入库 | 完成 | 分支 `research/qm-research-001` |
| Drive 归档 | 进行中/完成见回复 | `deliverables/` + 研究目录 |

## 2. 累计预算

| 项 | 用量 | 上限 |
|---|---|---|
| 策略账户路径 | 350（含压力/借券情景） | 600 |
| 正式计算墙钟 | 约 431 秒 + 复跑 53 秒 | 8 小时 |
| CPU | 单进程 | 4 核 |
| 数据采购 | 0 | 0（未新增订阅） |

## 3. 精确恢复命令

```bash
cd ~/projects/trading/Vibe-Trading/research/qm_research_001
V=~/projects/trading/Vibe-Trading/.venv/bin/python
$V scripts/run_all.py      # code/data hash 一致则跳过已完成 350 项
QM_FORCE=1 $V scripts/run_all.py   # 强制重算
$V scripts/report.py && $V scripts/funnel.py
```

若 `panel.npz` 或 `edgar_8k202.csv` 缺失，先跑 `scripts/build_panel.py` 与 `scripts/crawl_edgar.py`（约 39 秒 / 16 分钟）。

## 4. 唯一下一步

补 2016–2026 美股 1 分钟数据（含成交额），只对 EP 的 583 个跳空事件复现方案原版（OR5 + 09:35–09:59 累计量 + 下一分钟成交）。
需 Jake 先决定是否采购数据（本轮未购买任何订阅）。

## 5. 阻塞项（不重试）

- 历史分钟线：免费渠道均不可得（Yahoo 30 天上限、Stooq 拒绝、无 Databento 个股分钟缓存）。
- 历史借券/费率/召回/SSR：无来源。
- 含退市证券的历史股票池：无来源，生存偏差未消除。
- EP-C2：无 point-in-time 财务面板，未运行。
