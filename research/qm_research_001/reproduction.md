# QM-RESEARCH-001｜复现（reproduction）

## 1. 环境

| 项 | 值 |
|---|---|
| 机器 | MacBook 本地（macOS 27.0.1），未使用云算力 |
| 解释器 | `~/projects/trading/Vibe-Trading/.venv/bin/python`（Python 3.12.13） |
| 依赖 | pandas 2.3.3、numpy 2.5.4、scipy 1.18.1、duckdb 1.5.6、yfinance 1.7.0 |
| 数据（只读） | `/Users/jake/projects/trading/data/ts_live/raw/*.parquet`（112 分片，2006–2026-10-20） |
| 数据（事件） | SEC EDGAR `data.sec.gov/submissions`（8-K Item 2.02） |
| 代码仓库 | github.com/jakeguo117/Vibe-Trading → `research/qm_research_001/` |

## 2. 命令

```bash
cd ~/projects/trading/Vibe-Trading/research/qm_research_001
V=~/projects/trading/Vibe-Trading/.venv/bin/python

# 1) 面板（39s，4,477 交易日 × 10,967 证券，680MB）
$V scripts/build_panel.py runs/qm-001-20261022-0041/data

# 2) 引擎自检（13/13）
$V tests/test_engine.py

# 3) 财报事件（首次约 16 分钟，SEC 限速；已落盘可复用）
$V scripts/make_ep_universe.py
$V scripts/crawl_edgar.py runs/qm-001-20261022-0041/data/ep_syms.txt \
     runs/qm-001-20261022-0041/data/edgar_8k202.csv \
     runs/qm-001-20261022-0041/data/cik_map.json

# 4) 冻结配置与评估清单
$V scripts/freeze.py

# 5) 批处理（211 条评估，约 64s；断点续跑）
$V scripts/run_all.py
QM_FORCE=1 $V scripts/run_all.py     # 强制重算

# 6) 报告
$V scripts/report.py
$V scripts/funnel.py
```

## 3. 指纹

| 对象 | sha256 |
|---|---|
| 方案原文 | 见 `runs/qm-001-20261022-0041/config/config.json` → `spec_sha256` |
| 面板 `panel.npz` | 见 `config.json` → `data_hash_panel`，完整值 `artifact_checksums.txt` |
| EDGAR 事件 CSV | 见 `config.json` → `data_hash_edgar` |
| 代码 | 见 `config.json` → `code_hash`（`src/qm/*.py` 全量哈希前 16 位） |

## 4. 断点与接续

- `checkpoint.json` 记录每个 `(策略|配置|分段|code_hash|data_hash)` 的指标与累计预算（paths、wall_s）。
- 恢复时若 code_hash 与 data_hash 一致，已完成项直接跳过，不重复计算；不一致则重算并在新 key 下记录。
- 每个路径的结果先写 CSV/JSON 到 `deliverables/<策略>/<配置>/`，再写 checkpoint；中断最多丢当前一项。
- 本次实测：首次跑完后再次启动 `run_all.py`，全部路径被跳过（`budget.paths` 不增加），仅基准重算后退出。

## 5. 已知不可复现项

- 分钟级结果：无数据，任何声称复现的方案都不可信。
- 借券成本：只有 10/30/100% 假设情景，不是历史费率。
- 生存偏差：股票池为当期快照，重跑也不会改变。

## 6. 目录对应

| 交付物 | 位置 |
|---|---|
| 三策略结论 | `00_summary_zh.md` |
| 数据审计 | `data_readiness.md`、`runs/.../data/coverage.csv` |
| 冻结配置 | `spec_frozen.md`、`runs/.../config/{config,manifest}.json` |
| 自检 | `test_evidence.md`、`tests/test_engine.py` |
| 比较/压力/扰动 | `runs/.../deliverables/{comparisons,stress,borrow_scenarios}.csv` |
| 验收 | `runs/.../deliverables/acceptance.csv` |
| 统计/归因 | `statistics.md`、`attribution.md`、`runs/.../deliverables/{statistics,attribution}.csv` |
| 逐日净值/逐笔/拒绝/漏斗 | `runs/.../deliverables/<策略>/<配置>/<分段>_{equity,trades,rejects}.csv`、`funnel.csv` |
| 断点/日志 | `runs/.../checkpoint.json`、`runs/.../run.log` |
| 哈希 | `artifact_checksums.txt` |
