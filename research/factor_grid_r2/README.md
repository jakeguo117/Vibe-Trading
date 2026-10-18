# 因子网格第二轮（稳健性与衍生品）

这是研究代码，不是实盘下单程序。交易系统仓库的第 13 号拉取请求从本环境读不到，所以规则按任务说明和正本 v1.1.1 引擎重写，跑在一份固定的流动性名单上，不是全市场扫描。名单有幸存者偏差，结论不能直接拿去交易。

门槛写在 `thresholds.json`，程序只读不改。价格缓存在 `cache/`，不进版本库。随机种子 20261013。

```bash
python3 research/factor_grid_r2/run.py
```

输出在 `research/factor_grid_r2/results/`：`all.csv`、`summary_zh.md` 和图表。
