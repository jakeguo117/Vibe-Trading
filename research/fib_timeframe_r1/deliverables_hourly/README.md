# fib_timeframe_r1 — Fib 贡献与交易周期缩短（研究实现）

对应方案：`Jake_Fib_Timeframe_Test_Plan_v0.1_20261020.md`
（sha256 `ca25225fed75769c7454b81b752f40c184a9c4a9f614c9233068f7e79365f0a5`）
研究编号：JAKE-FIB-TIMEFRAME-20261020-v0.1

实现口径与差异记录见 [SPEC.md](SPEC.md)；结论与全部附表见 [deliverables/REPORT_ZH.md](deliverables/REPORT_ZH.md)。

## 状态

**r2（2026-10-20，小时批）**：18 个核心配置全部 **DONE**（6 个此前 `BLOCKED_NO_HOURLY_DATA` 已用
购买的 Databento `ohlcv-1h` 解锁）；15 个主比较全部有结果，**仍无一通过方案 §10.2 的 7 条门槛**。
小时批的结论见 [deliverables_hourly/REPORT_ZH.md](deliverables_hourly/REPORT_ZH.md)
（r1 报告见 [deliverables/REPORT_ZH.md](deliverables/REPORT_ZH.md)，A 线与 W/D 结论以它为准）。

## 依赖

- 只读参考仓库 `trading-system`，提交 `2ac825b56c9121f7bb4616c4e5ab18a2619df657`
  （`aport.py` 是它 `backtest/upleg_bt/engine/portfolio.py` 的逐行副本，blob `488ba770…`）
- 正式锁定快照 `data/ts_v111`（`mat/`、`raw/`、`cache/signals_primary.npz`）
- Python 3.12，numpy / pandas / pyarrow（版本记录在 `results/preflight.json`）

## 运行顺序

```bash
export SNAP=/path/to/data/ts_v111
export TS=/path/to/trading-system          # 只读
export PYTHONPATH=src:$TS/backtest

python -m upleg_bt reproduce v1.1.1                                    # Gate 0（在 $TS 下跑）
python -m fib_r1.adriver  --out runs/a_line --data-dir $SNAP --backtest-dir $TS/backtest
python -m fib_r1.acost    --out runs/a_line --data-dir $SNAP --backtest-dir $TS/backtest
python -m fib_r1.aepisode --out runs/a_line --data-dir $SNAP --backtest-dir $TS/backtest
python -m fib_r1.cdriver  --out runs/c_line --raw-dir $SNAP/raw --comm-dir $SNAP/commodities --stress
python -m fib_r1.bdriver  --out runs/b_line --raw-dir $SNAP/raw --backtest-dir $TS/backtest --stress
python -m fib_r1.bxp      --out runs/b_line --raw-dir $SNAP/raw --backtest-dir $TS/backtest
python -m fib_r1.extra    --out runs/extra  --raw-dir $SNAP/raw --run-root runs
python -m fib_r1.prereg   runs/meta                                    # preflight + preregistration
python -m fib_r1.compare  runs runs/comparisons.csv
python -m fib_r1.aggregate runs
python -m fib_r1.render   runs                                         # REPORT_ZH.md
```

## 测试

```bash
PYTHONPATH=src python -m pytest tests -q
```

覆盖方案 §11 的 C 线部分：不得前视、突破窗口不含当根、收缩分位边界（24/120 与 25/120）、
不得同 bar 成交、跳空按实际开盘而非止损线、单持仓、现金非负、费用不重复、确定性重跑、
延迟不得把已冻结的退出单推后、整数股与最低手续费。
