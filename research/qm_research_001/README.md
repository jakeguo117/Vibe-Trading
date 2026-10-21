# QM-RESEARCH-001｜Qullamaggie 三类策略独立研究

研究代码与运行产物目录。方案正本在 `spec/`，数据审计在 `data_readiness.md`。

## 结论速览

- 主方案要求的**历史分钟线不可获得**（Yahoo 仅最近 30 天；Stooq 拒绝；Databento 本地只有 7 个 ETF 的小时线），
  因此 OR5 + 下一分钟成交无法复现，主结论状态 **DATA_INCOMPLETE**。
- **历史借券/费率/召回/SSR 不可获得** → QM-PS-001 只能到 PS-PRICE，执行状态 INCOMPLETE。
- **历史股票池只有当期上市公司**（10,967 个中仅 13 个在窗口内停止交易）→ 生存偏差，绝对收益偏高。
- 已交付：可运行引擎 + 13 项经济行为自检 + 三类策略的 **EOD 诊断变体**（明确命名，不替代主方案）。

## 目录

```
spec/                 方案原文（下载快照，sha256 见 config.json）
data_readiness.md     数据审计（含证据与状态）
src/qm/               core(面板/筛选) engine(账户内核) signals metrics
scripts/              build_panel / crawl_edgar / freeze / run_all / report / drive_api
tests/test_engine.py  13 项合成数据自检
runs/qm-001-20261022-0041/
  data/               panel.npz, coverage.csv, edgar_8k202.csv, 核验 JSON
  config/             config.json, manifest.json（评估清单）
  deliverables/            comparisons.csv, acceptance.csv, stress.csv, statistics.csv,
                      benchmarks/, QM-BO|EP|PS/<config>/<segment>_{equity,trades,rejects}.csv
```

## 复现

```bash
cd ~/projects/trading/Vibe-Trading/research/qm_research_001
V=~/projects/trading/Vibe-Trading/.venv/bin/python
$V scripts/build_panel.py runs/qm-001-20261022-0041/data      # 39s，只读 ts_live 快照
$V tests/test_engine.py                                        # 13/13
$V scripts/freeze.py                                           # 冻结 config + manifest
$V scripts/run_all.py                                          # 批处理 + checkpoint
QM_FORCE=1 $V scripts/run_all.py                               # 强制重算
$V scripts/report.py                                           # 验收/统计/归因
```

## 重要限制

1. 全部结果为 **EOD 诊断**，不是方案 §3.3/§4.3/§5.2 的盘中执行版本。
2. 股票池生存偏差未消除。
3. 无停牌、无借券、无历史财报数值面板；EP-C2 未运行。
4. 未修改 `trading-system`，未下单，未新增付费数据。
