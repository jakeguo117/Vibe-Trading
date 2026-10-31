# Fixed20 烤炉壳

同一批 20 只上的三条腿：等权买入持有（BH）、缩短 Fib（F）、双动量加破位（N1）。

这个目录只用合成的假价格，把三条腿的壳接上。

**不出绩效表。不跑真窗。** 不写 `summary.csv`、`relative.csv`、`equity_*.csv`、`nav_compare.png`。不跑 2019-01-02 到 2025-10-30。真窗要等以后单独开闸。这里不改实盘，不改每日信号。

数字以 Fixed20 Freeze v1 为准。和 Spec 打架时听 Freeze。

## 放在哪

代码在仓库根下的 `bakeoff/fixed20/`。产品回测在 `agent/backtest`，那是另一套，这次不放进去，也不改它。

## 怎么测

在仓库根目录：

```bash
python -m pytest bakeoff/fixed20/tests -q
```

合成小窗（假价格，不是历史成绩）：

```bash
python -c "from bakeoff.fixed20 import run_synthetic; run_synthetic()"
```

没装上 `upleg_bt` 时，上面这句会直接失败，不会拿另一套公式顶上。返回值只活在内存里。打印出来的净值不是 Fixed20 真窗数字。

## 时点

时点接口在本目录 `pit.py`。不引用 trading-system 的 `research.data`。

决策日是 D 时，`delay=1` 的可知日是日历上的前一天。那天之后的 bar 不给。信号查询用 `delay=0`，因为 Freeze 里的 delay=1 指的是下一交易日才成交。若信号再藏一天，执行会晚两天。成交仍是下一根 bar 的开盘。三腿同一个 `t0`、同一个首执日。

夹具里的 open/close 当作复权开盘和复权收盘。初始净值 1。不满仓时现金闲着。不做空，不做波动目标。

## F 腿要的生产代码

排序调用 `upleg_bt.engine.fm.FM.score`，格子累计调用 `FM.cells`。阶段调用 `upleg_bt.engine.fib_locked.fib_stages_weekly`。费用调用 `upleg_bt.costs.trade_fee`。

钉死的提交：`70038a53e7992b49f10e5906b18573a2beddb965`

安装命令在 `requirements-upleg.txt`。只读安装，不把算法抄进本仓库。

本环境拉取结果：失败。GitHub 返回 `Repository not found`。当前云代理的凭证读不到私有仓库 `jakeguo117/trading-system`。没有改用自写 Fib 或自写胜率。

要跑通 F 腿，需要下面任意一种：

1. 给云代理一个能读 `jakeguo117/trading-system` 的 token，再按 `requirements-upleg.txt` 安装。
2. 把该提交加成本仓库的只读 submodule，让 `upleg_bt` 出现在 Python 路径上。提交号不变。

装好之前：要 Fib、hist 或生产费率的测试会跳过。BH 和 N1 的目标权重测试不依赖这些函数，照常跑。三条腿一起撮合的测试也会跳过，因为手续费同样来自 `upleg_bt.costs`，不能另写一套费率。

合成夹具没有已经结束的主升样本。这时 `FM.score` 的原逻辑是整体均值 0，再减去两倍单边成本。这不是另一套胜率公式。真出数之前，必须把生产的 segments 送进这同一个函数。

上涨段用 50 日 SMA，不再用生产里的 30 周。k=3。只有主升可以新开。鱼尾只维持、不开新仓。磨底和震荡不开新仓。离开上涨段先减半一次。连续 3 个信号日处于下跌确认则清掉。掉出前 k 也清掉。硬止损是本波段成本 −8%，优先于减半。

Fixed20 里若 GLD 和 USO 同时进池，沿用生产顺序 GLD → SLV → USO。只进一个时按 hist 排序，不单独插队。

## 壳上写死的几件事

- 宇宙顺序在 `universe.py`。资格日 2018-12-31 只作文档，不再筛流动性。
- A1：单一主键 `META`。`<=2022-06-08` 用当时的 FB 价量，`>=2022-06-09` 用 META。`splice_meta_fb` 接成一条序列。合成夹具里已经是连续 META，没有单独的 FB 行。
- A2：只用 GOOGL。
- 成本走 `upleg_bt.costs`（5 bp / 10 bp，2000 万美元分界）。GLD、USO 沿用生产里商品的流动性费率。费率字段在返回的 meta 里。不允许零成本捷径。

## N1

`ret_20` 加收盘不低于 SMA50，取前 k=3，等权。跌破均线或掉出前 k 直接清仓，不减半。硬止损同样是 −8%。不用 Fib，不用 hist。

## BH

`t0` 当天有 bar 的名字等权，下一交易日一次买完，之后不调权。`t0` 缺 bar 的名字权重为 0，其余名字把满仓分完，以后也不补。

## 不做

- Yahoo / Drive 真窗，以及 Freeze 第 4 节的绩效表
- 研究池周交接
- 并行 sleeve、双账本
- 换 v1.1.2 或每日信号主路径
