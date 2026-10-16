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
    skipped = payload.get("skipped") or []
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
    skip_lines = "\n".join(f"- 战术{i}：{reason}" for i, reason in skipped)
    top2 = payload.get("top2") or []
    top2_text = "、".join(f"{n}（年化 {_pct(a)}）" for n, a in top2) if top2 else "—"
    cov = payload["coverage"]
    start = payload.get("start", "2016-01-01")
    text = f"""# A股短线战术全量日线回测（{start} 起）

本文件是研究结果，不是交易建议。各战术各自独立建仓，互不混合。起始资金 100 万元，最多同时持有 10 只，每只目标仓位为当时权益的 10%。信号多于空位时，按信号前 20 个交易日平均成交额从高到低占用名额。

## 数据

- 区间：{start} 至 {payload["end"]}。指标用 2015 年起的日线预热，预热段不成交。
- 行情：AkShare 东方财富 `stock_zh_a_hist`（Hermes 可达；此前云端代理因 TLS 被断开改用新浪）。未复权 OHLCV + `factor = 未复权收盘 / 前复权收盘`。沪深300 / 中证全指用 AkShare 指数日线（中证全指必要时走 `stock_zh_index_daily_tx`）。
- 前复权价格只用于均线、MACD、KDJ、SAR、CCI 和涨幅判断；下单股数、涨跌停和成交额用未复权价格。
- 费用：佣金万 2.5、单边最低 5 元；卖出印花税在 2023-08-28 之前为 0.1%，当日及之后为 0.05%；买卖各计 0.1% 滑点。
- 交易规则：T+1；买入 100 股整数倍；涨停开盘不成交，跌停开盘顺延到下一根可卖开盘；成交量为 0 的停牌日已剔除。主板 10%；创业板 2020-08-24 前 10%、后 20%；科创板 20%；北交所 30%。最新简称含 ST 的按 5% 涨跌停处理。
- 样本：当前 A 股 {cov["spot"]} 只；上交所退市并入 {cov.get("sse_delist_added", 0)} 只（有行情 {cov.get("sse_delist_bars", 0)}）；深交所退市并入 {cov.get("szse_delist_added", 0)} 只（有行情 {cov.get("szse_delist_bars", 0)}）。成功写入日线 {cov["bars"]} 只，失败 {cov["failed"]} 只。2016 年 1 月有行情的 {cov.get("present_jan_2016", "—")} 只。
- 本次运行耗时约 {cov.get("runtime_sec", "—")} 秒。

## 跳过的战术

{skip_lines}

## 已测战术的编码默认

形态类战法在笔记未给出明确离场时，统一默认：**前复权涨幅 +15% 止盈**，或收盘跌破 MA5 后下一开盘卖出，或最多持有 10 个交易日（买入当日为第 1 日）。战术16 用跌破 MA10；战术12/26 用更长均线（见下）。战术5/13/19/25 仍按原笔记规则（与 PR #4 引擎一致）。

1. 战术2 涨停回调：横盘≥20 日后续涨停，回调约一半涨幅后放量阳线≥3% 次日开盘买。
2. 战术3 天衣无缝：底部一字/T 字突破，次日高开 2%–5% 且非巨量，当日开盘买。
3. 战术4 反包上影：倒锤后数日内收盘吞没上影，且 MA5>10>20，次日开盘买。
4. 战术5 双均线：MA5 上穿 MA20，次日收盘确认，第三日开盘买；+8% 或跌破 MA5。
5. 战术7 蛟龙出海：MA5/10/30 粘合，放量大阳穿过并收在三条之上，次日开盘买。
6. 战术8 一字涨停：低位横盘后一字板，次日开盘买（非涨停开盘）。
7. 战术9 飞雁开路：涨停后 ±3% 缩量整理，再放量突破，次日开盘买。
8. 战术12 CCI 月线：月 CCI 自 −100 上穿 +100，零轴上两底后，日线突破 MA5 买；跌破 MA20 或满 40 日卖。
9. 战术13 主力521：MA5 上穿 MA21 且倍量，次日阳线涨幅>5%，再下一开盘买；跌破 MA5 或满 20 日。
10. 战术16 蚂蚁上树：五连小阳后放量突破 MA10，回调阴线次日买；跌破 MA10 卖。
11. 战术17 擒龙捉妖：股价<10 元、涨停、站上 MA5、均线多头、放量，次日开盘买。
12. 战术19 抛物转向：SAR(4,2%,2%,6%) 翻转且前段≥5 根，次日开盘买卖。
13. 战术21 量价反包：前日 J<20，当日价量双吞没且收盘>MA5，次日开盘买。
14. 战术22 涨停三连阴：非一字涨停→次日高开→三根不破涨停开盘价的缩量小阴，信号次日开盘买。
15. 战术23 双阳定乾坤：倍量涨停→约 5 日缩量洗盘→更高位置第二涨停，次日开盘买。
16. 战术24 三连阴涨停：实体涨停→次日新高→缩量阴→放量阳反包，次日开盘买。
17. 战术25 双指标：DIF>0 且升、KDJ 金叉、收盘>MA20、未复权价<6、非 ST；死叉或跌破 MA20 卖。
18. 战术26 双阳单阴：站上 MA60 的倍量大阳（涨幅≥5%）后缩量单阴不破平台/MA60，阴线次日开盘买；跌破 MA60 卖。

基准是沪深300（000300）和中证全指（000985）价格指数买入持有，不加费用。

## 结果

无风险利率取 0。年化按 252 个交易日。Sharpe 用样本标准差（ddof=1）。Sortino 用全部收益日上的下行均方根。

表现最好的两套战术（相对年化）：{top2_text}。对比图见 `top2_vs_csi300.png`（沪深300 为黑色粗线，最多三根线）。

{table}

分年收益：

{year_table}

战术5 按入场时 MA20 斜率拆开（已成交交易）：

{slope_table}

## 幸存者偏差

本次已并入上交所与深交所（Hermes 上 SZSE 接口可用，云端当时 TLS 失败）2015 年后终止上市名单。2016 年 1 月有行情 {cov.get("present_jan_2016", "—")} 只。退市股已实现盈亏：{payload["delist_pnl"]}。

A 股 2020 年前退市很少，2016–2019 的幸存者高估应明显小于美股常见水平；2020 年后退市加快且集中在低价/ST，对战术17/25 等低价过滤策略更敏感。表中收益应视为偏乐观的上沿。ST 过滤用最新简称，存在前视偏差；「战术25（不做ST过滤）」用于敏感性对照。

原始日线只在本地 `data-cache`，不进入仓库。云端 PR #4 对战术5/13/19/25 在 2015–2026 的结论（四套均大幅跑输）与本表同向，本跑以 2016 起点做全量对齐。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
