"""Chinese write-up. Numbers are filled from the run; the rules are not."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.2f}%"


def _num(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def _md_table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
    show = frame.copy()
    for col, kind in formats.items():
        if col not in show.columns:
            continue
        if kind == "pct":
            show[col] = show[col].map(_pct)
        elif kind == "int":
            show[col] = show[col].map(lambda v: "—" if pd.isna(v) else str(int(v)))
        elif kind == "num":
            show[col] = show[col].map(lambda v: _num(v))
        elif kind == "days":
            show[col] = show[col].map(lambda v: "—" if pd.isna(v) else f"{v:.1f}")
    header = "| " + " | ".join(show.columns) + " |"
    rule = "| " + " | ".join("---" for _ in show.columns) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in show.itertuples(index=False))
    return "\n".join([header, rule, body])


def write_summary(path: Path, payload: dict) -> None:
    """Write ``summary_zh.md`` from the finished tables and coverage counts."""
    summary: pd.DataFrame = payload["summary"]
    yearly: pd.DataFrame = payload["yearly"]
    slope: pd.DataFrame = payload["slope"]
    labels = {
        "strategy": "策略",
        "annualized_return": "年化收益",
        "max_drawdown": "最大回撤",
        "sortino": "Sortino",
        "sharpe": "Sharpe",
        "win_rate": "胜率",
        "avg_holding_days": "平均持有天数",
        "trade_count": "成交笔数",
        "total_return": "累计收益",
        "final_equity": "期末权益",
    }
    show = summary.rename(columns=labels)
    table = _md_table(
        show,
        {
            "年化收益": "pct",
            "最大回撤": "pct",
            "Sortino": "num",
            "Sharpe": "num",
            "胜率": "pct",
            "平均持有天数": "days",
            "成交笔数": "int",
            "累计收益": "pct",
            "期末权益": "num",
        },
    )
    year_table = _md_table(yearly, {col: "pct" for col in yearly.columns if col != "年份"})
    slope_show = slope.rename(
        columns={
            "slope": "MA20斜率",
            "trade_count": "笔数",
            "win_rate": "胜率",
            "avg_return": "平均单笔收益",
            "avg_holding_days": "平均持有天数",
            "total_pnl": "盈亏合计(元)",
        }
    )
    slope_table = _md_table(
        slope_show,
        {"胜率": "pct", "平均单笔收益": "pct", "平均持有天数": "days", "盈亏合计(元)": "num", "笔数": "int"},
    )
    ann = summary.set_index("strategy")["annualized_return"]
    tactic_names = [name for name in ann.index if str(name).startswith("战术") and "不做" not in str(name)]
    win = summary.set_index("strategy")["win_rate"]
    if tactic_names and all(float(ann[name]) < 0 for name in tactic_names):
        win_clause = "胜率都低于一半，" if all(float(win[name]) < 0.5 for name in tactic_names) else ""
        reading = (
            "四套战术的年化收益都是负的，最大回撤接近亏光本金。"
            f"{win_clause}持有期只有数日，"
            "滑点、佣金和卖出印花税会在每一轮里重复扣除；扣费后的边际为负，账户越做越小，"
            "后期权益不够买一手时曲线走平。同一区间里沪深300和中证全指的年化收益是正的。"
        )
    else:
        reading = "各战术与两个价格指数的对比见下表。年化按 252 个交易日、无风险利率取 0。"
    cov = payload["coverage"]
    text = f"""# A股四套短线战术回测（2015年起）

本文件是研究结果，不是交易建议。四套战术各自独立建仓，互不混合。起始资金 100 万元，最多同时持有 10 只，每只目标仓位为当时权益的 10%。信号多于空位时，按信号前 20 个交易日平均成交额从高到低占用名额。

## 数据

- 区间：2015-01-01 至 {payload["end"]}。指标用 2014 年起的日线预热，预热段不成交。
- 行情：AkShare 的新浪日线（`stock_zh_a_daily` 同一接口：`hisdata_klc2` 不复权 K 线 + `qfq.js` 前复权因子）。东方财富 `stock_zh_a_hist` 在本次运行环境中连接被对端直接断开，因此没有用它。沪深300用新浪指数日线；中证全指的新浪序列停在 2016 年，改用 AkShare 的 `stock_zh_index_daily_tx` 补全到最新交易日。
- 前复权价格只用于均线、MACD、KDJ、SAR 和涨幅判断；下单股数、涨跌停和成交额用未复权价格。前复权定义是 `qfq = 未复权价 / 因子`（因子在最近一次除权后为 1）。持仓市值按 `股数 × 未复权价 × 入场因子 / 当日因子` 计，卖出时把因子差记为分红或拆分的现金残差，避免把除权当成亏损。有 112 只股票的因子在某一天跳变，但未复权价几乎不动（前复权价会出现 30% 以上的假缺口）。这些跳变已按「未复权价不动则因子也不跳」修正，真正的拆分（价格和因子同比例变化）保留。
- 费用：佣金万 2.5、单边最低 5 元；卖出印花税在 2023-08-28 之前为 0.1%，当日及之后为 0.05%；买卖各计 0.1% 滑点。未另加过户费。
- 交易规则：T+1；买入 100 股整数倍；涨停开盘不成交，跌停开盘顺延到下一根可卖开盘；成交量为 0 的停牌日已从序列中剔除。主板 10%；创业板在 2020-08-24 之前 10%、之后 20%；科创板 20%；北交所 30%。名称中含 ST 的股票按 5% 涨跌停处理（见下文偏差）。科创板、注册制后的创业板和北交所上市前 5 个交易日不设涨跌停。
- 样本：新浪当前沪深京 A 股 {cov["spot"]} 只，另加入上交所 2015 年以来终止上市且不在当前列表中的 {cov["sse_delist_added"]} 只。成功写入日线 {cov["bars"]} 只，失败 {cov["failed"]} 只。2015 年 1 月有行情的 {cov["present_jan_2015"]} 只。

## 战术

1. 战术5 双均线：T 日 MA5 上穿 MA20，T+1 收盘高于 T 日收盘，T+2 开盘买入。止盈为相对买入当日开盘的前复权涨幅达到 8%（笔记里 5%–10% 的中点，没有再扫参数），否则收盘跌破 MA5 后下一开盘卖出。入场前一根 K 线的 MA20 相对 5 日前变化超过 +0.5% 记为向上，低于 −0.5% 记为向下，其余记为走平。
2. 战术13 主力521：T 日 MA5 上穿 MA21 且成交量不低于前一日的 2 倍；T+1 为阳线且收盘涨幅大于 5%。下一开盘买入。收盘跌破 MA5 则下一开盘卖出，最多持有 20 个交易日（买入当日为第 1 日）。
3. 战术19 抛物转向：SAR 参数按笔记「4,2,2,6」实现为周期 4、步长 2%、增量 2%、极限 6%。收盘确认价格上穿 SAR 后下一开盘买入，下穿后下一开盘卖出。上一段趋势短于 5 根 K 线的翻转信号忽略，不买也不卖。
4. 战术25 双指标：MACD(12,26,9) 的 DIF 大于 0 且继续上升，KDJ(9,3,3) 的 K 上穿 D，收盘高于 MA20，未复权收盘价低于 6 元，且最新简称不含 ST。下一开盘买入。K 下穿 D 或收盘跌破 MA20 后下一开盘卖出。KDJ 用国内 SMA，初值 50。

基准是沪深300（000300）和中证全指（000985）的价格指数买入持有，不加费用、不含股息。策略权益含前复权所代表的分红残差，所以相对价格指数的超额里有一部分是股息口径差异，不是纯选股超额。

## 结果

风险指标的无风险利率取 0。年化按 252 个交易日。Sharpe 用样本标准差（ddof=1）。Sortino 用全部收益日上的下行均方根，与仓库里 `backtest.metrics` 的定义一致。

{reading}

{table}

分年收益（当年年末权益 / 上一年年末权益 − 1；2015 年相对起始资金）。账户缩水之后，后期的百分比是在很小的权益上算的，不表示本金被赚回：

{year_table}

战术5 按入场时 MA20 斜率拆开的是**已经成交**的交易，不是三套独立组合：

{slope_table}

净值图每个战术一张，最多三根线：策略、沪深300（黑色粗线）、中证全指。图例放在坐标轴上方。

## 幸存者偏差

国家统计局口径的境内上市公司数（A、B 股）2014 年末 2613 家、2015 年末 2827 家、2020 年末 4154 家、2023 年末 5107 家（中研网转载的年度表）。中国上市公司协会统计、新华网 2025-01-28 报道：2024 年末 5392 家，当年退市 54 家。新浪财经 2024 年末盘点给出的退市家数是 2020 年 19、2021 年 22、2022 年 46、2023 年 46、2024 年 52，其中 2024 年有 38 家是面值退市。财新 2024-04-13 写到，2020 年以来已有 135 家退市。

本次缓存里，2015 年 1 月能对上行情的有 {cov["present_jan_2015"]} 只，对照 2014 年末约 2613 家上市公司。差额主要来自：深交所终止上市名单（交易所接口在本环境 TLS 被断开，没能并入）、下载失败的 {cov["failed"]} 只、以及 B 股和代码口径差异。上交所 2015 年后终止上市的股票已尽量用新浪日线补回（加入 {cov["sse_delist_added"]} 只，其中写出行情 {cov["sse_delist_bars"]} 只）。

Shumway（1997, Journal of Finance, *The Delisting Bias in CRSP Data*）说明，丢掉退市收益会把平均收益做高，美股小盘上等权组合里这不是可以忽略的零头。A 股在 2020 年退市改革前退市很少，壳价值也使很多退出不是归零，所以 2015–2019 的高估应明显小于美股。2020 年之后退市加快，而且集中在低价、ST 股票：这对「收盘价低于 6 元」的战术25 比对其余三套更要紧。

用本次已经补回的上交所退市股衡量方向：这些代码贡献的已实现盈亏为 {payload["delist_pnl"]}。若这笔盈亏为负，只保留存活股会把收益做高；深市未能补回的退市股数量与沪市同一时期大致同级，残余高估大约是这一笔影响再放大一倍的量级，而不是把整个组合年化再加几个点的量级——因为组合同时只持有 10 只、持有期以数日到数十日计，退市股只有在发出信号时才会进组合。战术25 的信号本身就落在低价股上，残余高估会更大，表中的战术25 应看成偏乐观的上沿。

另外两项会推高结果、且没有在数字里扣掉的口径：

- ST 过滤和 5% 涨跌停用的是**最新简称**。新浪曾用名和东财 F10 都只给名称序列、不给变更日期，做不出时点 ST。用今天的 ST 去剔除整段历史，会提前丢掉后来才被 ST 的股票，战术25 因此偏乐观。表里「战术25（不做ST过滤）」是关掉名称过滤后的同一套规则，用来看这个口径有多敏感。
- 基准是价格指数。策略把分红留在权益里，牛市里的超额会好看一些。

原始日线只在本地缓存，不进入仓库。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
