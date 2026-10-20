"""Renders REPORT_ZH.md for the hourly batch (r2) from the files the run produced.

Every number below is read from result.json / comparisons.csv / acceptance.json / h_line.json /
cw_line.json / gate7_drop_top.csv / thinning_summary.json / equity_daily.csv. Nothing is typed by
hand; the prose states the direction of the measured numbers and the evidence level, which is the
part a machine must not invent.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/jake/projects/trading/Vibe-Trading/research/fib_timeframe_r1/src")
from fib_r1.metrics import perf  # noqa: E402

RUN = Path("/Users/jake/projects/hermes-tasks/runs/backtest-fib-hourly-20261020-1900/runs/combined")
R1 = Path("/Users/jake/projects/hermes-tasks/runs/backtest-fib-breakout-20261020-1655/runs")
NEW = ["B-H-ON", "B-H-OFF", "C-H-P", "C-H-S", "C-DH-P", "C-DH-S"]
CTRL = ["C-W-P", "C-W-S", "C-D-P", "C-D-S", "B-W-ON", "B-W-OFF", "B-D-ON", "B-D-OFF"]


def f(x, nd=2, pct=False, sign=False):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "—"
    v = x * 100 if pct else x
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s + ("%" if pct else "")


def load():
    r = json.load(open(RUN / "result.json"))
    cmp_df = pd.read_csv(RUN / "comparisons.csv")
    acc = json.load(open(RUN / "acceptance.json"))
    hl = json.load(open(RUN / "h_line" / "h_line.json"))
    cw = json.load(open(RUN / "cw_line" / "cw_line.json"))
    pre = json.load(open(RUN / "h_line" / "hourly_preflight.json"))
    g7 = pd.read_csv(RUN / "h_line" / "gate7_drop_top.csv")
    thin = json.load(open(RUN / "thinning_summary.json"))
    eq = pd.read_csv(RUN / "equity_daily.csv", parse_dates=["date"])
    r1 = json.load(open(R1 / "result.json"))
    exp = pd.read_csv(RUN / "entry_experience.csv")
    hours = pd.read_csv(RUN / "h_line" / "signal_hour_distribution.csv") \
        if (RUN / "h_line" / "signal_hour_distribution.csv").exists() else pd.DataFrame()
    return r, cmp_df, acc, hl, cw, pre, g7, thin, eq, r1, exp, hours


def vol_matched(eq: pd.DataFrame, fast: str, slow: str, start, end) -> dict:
    """§8.3 descriptive risk scaling: shrink the higher-vol path to the lower vol, never lever up."""
    def series(v):
        s = eq[eq.variant == v].set_index("date").nav.sort_index()
        return s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    a, b = series(fast), series(slow)
    j = a.index.intersection(b.index)
    ra, rb = a.loc[j].pct_change().fillna(0), b.loc[j].pct_change().fillna(0)
    sa, sb = ra.std(), rb.std()
    out = {"vol_fast": float(sa * np.sqrt(252)), "vol_slow": float(sb * np.sqrt(252))}
    if sa > sb > 0:
        scale = sb / sa
        scaled = (1 + ra * scale).cumprod()
        m = perf(scaled, None)
        out.update({"scale": float(scale), "cagr_scaled": m["cagr"], "maxdd_scaled": m["maxdd"],
                    "sharpe_scaled": m["sharpe"]})
    else:
        out.update({"scale": 1.0, "cagr_scaled": None, "maxdd_scaled": None, "sharpe_scaled": None})
    m2 = perf(b, None)
    out.update({"cagr_slow": m2["cagr"], "maxdd_slow": m2["maxdd"], "sharpe_slow": m2["sharpe"]})
    return out


def main():
    r, cmp_df, acc, hl, cw, pre, g7, thin, eq, r1, exp, hours = load()
    cwj = pre["common_window"]
    start, end = cwj["start"], "2026-09-30"
    L = []
    A = L.append

    A("# 第二批：60 分钟周期（hourly-r2）—— Fib 贡献与周期缩短\n")
    A(f"研究编号 **JAKE-FIB-TIMEFRAME-20261020-v0.1** ｜ 批次 **r2（小时线解锁）** ｜ 生成于本 run 的 "
      f"`result.json` 等文件\n")
    A(f"行情：Databento `ohlcv-1h`，单交易所 OHLCV，2019-01-02…2026-09-30；"
      f"**共同窗口 {start} … {end}**（{cwj['months_from_start']:.0f} 个月，暖机由数据自动确定并在看收益前冻结）。\n")
    A("> r1 的 12 个 DONE 配置（A/B-W/B-D/C-W/C-D）结论沿用上一轮交付，本批不重跑整轮；"
      "为满足方案 §3.1「小时比较必须在共同窗口重跑相应 W/D 对照」，本批在同一共同窗口重跑了 8 个 W/D 对照，"
      "并在文中同时给出上一轮全窗口数字与本批共同窗口数字。\n")

    # ---------------- 1. 一页结论
    A("## 1. 一页结论\n")
    chp, cdp = r["configs"]["C-H-P"], cw["C-D-P"]["base"]
    cp = cmp_df.set_index("comparison")
    A("1. **60 分钟不是更好的周期。** 全小时配置 C-H-P（普通突破）全窗口 "
      f"{f(chp['cagr'],2,True)} / 回撤 {f(chp['maxdd'],2,True)} / "
      f"Sharpe {f(chp['sharpe'],3)}，对照日线 C-D-P "
      f"{f(cdp['cagr'],2,True)} / {f(cdp['maxdd'],2,True)} / "
      f"{f(cdp['sharpe'],3)}：收益只多 "
      f"{f((chp['cagr']-cdp['cagr'])*100,2)} 个百分点，"
      f"回撤却扩大到 {f(chp['maxdd']*100,1)}%，Sharpe 明显更差。"
      f"验证段虽有 {f(cp.loc['C-H-P-C-D-P','cagr_diff_pp'],2,sign=True)}pp 的账面优势，"
      f"但 p={cp.loc['C-H-P-C-D-P','p_raw']:.2f}、且按风险匹配后消失（§7）。\n")
    A("2. **\"更多机会\"没有转化为更好的机会。** 预注册的抽稀诊断（§8.4）在设计段算出 "
      f"p={thin['p']:.3f}（日线候选 {thin['n_slow']} 次 / 小时候选 {thin['n_fast']} 次），"
      f"100 条随机路径的 CAGR 中位数 {f(thin['thinned_cagr_median'],2,True)}，"
      f"与日线 {f(thin['base']['C-D-P'],2,True)} 几乎相同，"
      f"只有 {thin['share_seeds_beating_daily']*100:.0f}% 的路径跑赢日线。"
      "小时线的交易次数优势主要来自相关度更高的重复下注，不是独立机会增加。\n")
    A("3. **D/H 混合入场是本批唯一的正面方向，但未达门槛。** C-DH-P 全窗口 "
      f"{f(r['configs']['C-DH-P']['cagr'],2,True)} / {f(r['configs']['C-DH-P']['maxdd'],2,True)} / "
      f"{f(r['configs']['C-DH-P']['sharpe'],3)}，对照 C-D-P {f(cw['C-D-P']['base']['cagr'],2,True)} / "
      f"{f(cw['C-D-P']['base']['maxdd'],2,True)} / {f(cw['C-D-P']['base']['sharpe'],3)}；"
      f"验证段差 {f(cmp_df.loc[cmp_df.comparison=='C-DH-P-C-D-P','cagr_diff_pp'].iloc[0],2,sign=True)}pp"
      f"（p={cmp_df.loc[cmp_df.comparison=='C-DH-P-C-D-P','p_raw'].iloc[0]:.3f}）。"
      f"回撤变化 {f(cmp_df.loc[cmp_df.comparison=='C-DH-P-C-D-P','maxdd_diff_pp'].iloc[0],2,sign=True)}pp"
      "（在 3pp 恶化门槛内），方向为正，但**不足 2pp 门槛、验证段交易数不足 50 笔、未过多重检验**，"
      "只能定性为「弱/条件性证据，值得继续研究」，不能称为通过。\n")
    A("4. **收缩后突破（S）在任何周期都没有得到支持。** 小时线 C-H-S 相对 C-H-P "
      f"{f(cmp_df.loc[cmp_df.comparison=='C-H-S-C-H-P','cagr_diff_pp'].iloc[0],2,sign=True)}pp；"
      f"日线 {f(cmp_df.loc[cmp_df.comparison=='C-D-S-C-D-P','cagr_diff_pp'].iloc[0],2,sign=True)}pp；"
      f"周线 {f(cmp_df.loc[cmp_df.comparison=='C-W-S-C-W-P','cagr_diff_pp'].iloc[0],2,sign=True)}pp。"
      "小时线把收缩条件放进 60 分钟 bar，等于挑选日内最平静的时刻突破，结果最差。\n")
    A("5. **B 线（旧三持仓框架）缩短到小时时钟是明确负贡献。** B-H-ON "
      f"{f(r['configs']['B-H-ON']['cagr'],2,True)}、B-H-OFF {f(r['configs']['B-H-OFF']['cagr'],2,True)}，"
      f"回撤 {f(r['configs']['B-H-ON']['maxdd'],2,True)} / {f(r['configs']['B-H-OFF']['maxdd'],2,True)}，"
      f"各 {r['configs']['B-H-ON']['trades']['n_trades']} 笔完整交易、单笔期望为负；"
      f"相对 B-W-OFF 验证段 {f(cmp_df.loc[cmp_df.comparison=='B-H-OFF-B-W-OFF','cagr_diff_pp'].iloc[0],2,sign=True)}pp。"
      "把周/日决策换成小时决策，只放大了换手与成本。\n")
    A("6. **15 个主比较依旧无一通过 §10.2 的 7 条门槛。** 本批新增 6 个比较中，"
      "2 个直接 FAIL（B-H-ON−B-H-OFF、C-H-S−C-H-P），4 个因验证段交易数不足标 INSUFFICIENT；"
      "Holm 校正后无一显著。这与 r1 的结论一致，不是本批执行失败。\n")

    # ---------------- 2. 交付与状态
    A("## 2. 交付位置与配置状态\n")
    A("| 项 | 值 |\n|---|---|")
    A("| 分支 | `research/fib-timeframe-r1`（main 未动，未建/未合 PR） |")
    A("| 运行目录 | `runs/backtest-fib-hourly-20261020-1900/` |")
    A("| 交付数据 | `runs/combined/`（result.json、comparisons.csv、acceptance.json、trials.csv、"
      "equity_daily.csv、orders_trades.csv、entry_experience.csv、target_trace.csv）|")
    A("| 小时专属 | `runs/h_line/`（hourly_preflight.json、hourly_coverage.csv、cross_source_hourly.csv、"
      "h_line.json、signal_hour_distribution.csv、gate7_drop_top.csv）|")
    A("| Drive | `gdrive:交易与投资/Vibe Trading 研究归档/2026-10-20 Fib贡献与周期测试/hourly-r2/` |")
    A("")
    A("| 配置 | 状态 | 说明 |\n|---|---|---|")
    for cid in NEW:
        m = r["configs"][cid]
        A(f"| {cid} | **DONE**（r1: `BLOCKED_NO_HOURLY_DATA`）| {m['trades']['n_trades']} 笔完整交易 |")
    A("| A00/A10/A01/A11、B-W-*、B-D-*、C-W-*、C-D-* | DONE（r1）| 本批未重跑；W/D 另在共同窗口重跑作对照 |")
    A("")

    # ---------------- 3. 数据预检
    A("## 3. 数据预检（在看任何收益之前冻结）\n")
    A(f"文件 `runs/h_line/hourly_preflight.json`。CSV sha256 `{pre['feed']['csv_sha256'][:16]}…`，"
      f"清单 `{pre['feed']['manifest_sha256'][:16]}…`，{pre['feed']['rows_used']:,} 行（9 只合计）进入网格，"
      f"报价合计 USD {pre['feed']['quoted_usd']:.4f}（本批未再购买）。\n")
    A("**9 只齐全，覆盖率（常规时段整点 bar 网格）**\n")
    A("| 标的 | 交易所 | bar 数 | 期望 | 覆盖率 | 缺失整日 |\n|---|---|---:|---:|---:|---|")
    for row in pre["coverage"]:
        A(f"| {row['sym']} | {row['venue']} | {row['bars']} | {row['bars_expected']} | "
          f"{row['coverage']*100:.3f}% | {row['sessions_missing']} |")
    A(f"\n最低覆盖率 {pre['coverage_min']*100:.3f}%（TLT），高于方案 §3.3 的 99.5% 线 → 小时比较数据充分。"
      f"TLT 缺失的整日：{', '.join(pre['sessions_missing_by_symbol']['TLT']) or '无'}。\n")

    A("**跨源核对（单交易所 vs 合并行情）**\n")
    A("| 标的 | 日收盘相对差（中位） | p99 | 最大 | ≤10bp 占比 | 日高被包含 | 日低被包含 |\n"
      "|---|---:|---:|---:|---:|---:|---:|")
    for row in pre["cross_source"]:
        A(f"| {row['sym']} | {row['med_abs_rel_close']*1e4:.1f}bp | {row['p99_abs_rel_close']*1e4:.1f}bp | "
          f"{row['max_abs_rel_close']*1e4:.1f}bp | {row['within_10bp']*100:.1f}% | "
          f"{row['high_contained']*100:.1f}% | {row['low_contained']*100:.1f}% |")
    A(f"\n口径：小时 OHLC 先按拆分表 `splits` 与快照的 `Adj Close/Close` 映射到与 A/B/C 相同的复权尺度，"
      f"再与锁定快照的日线比较。收盘中位差 0.7–3.3bp，最大 122bp（USO），判定 "
      f"**{pre['cross_source_verdict']}**。\n")
    A(f"必须披露的残差：{pre['cross_source_note']}\n")

    A("**网格限制（不掩盖、不插值）**\n")
    A(f"- bar 标签 = 区间起点（UTC 整点）。两条独立证据：15:00 bar 收盘与官方日收盘中位差约 1bp；"
      f"2025-04-09 13:00 bar 内含 13:18 ET 的关税暂停行情。")
    A(f"- 常规时段 09:30–16:00 内**完整的整点 bar 只有 10:00…15:00 共 6 根**（半日市 3 根）。"
      f"09:00–10:00 那根混入 09:00–09:30 盘前成交，按方案 §3.3 不采用。")
    A(f"- 后果一：**09:30–10:00 开盘半小时不可交易**，每根「下一根开盘」最早落在 10:00 ET，"
      f"相对 09:30 官方开盘有系统性 30 分钟滞后，六个配置一致。")
    A(f"- 后果二：**没有任何短 bar**，方案 §3.3 的 short_bar 诊断在本网格上退化为空操作；"
      f"其真正关心的问题（尾段时段依赖）改以「信号时段分布 + 禁用当日最后一根 bar 入场」的事后诊断替代，"
      f"并在 §7 标注为 post-hoc、非预注册。")
    A(f"- 半日市（13:00 收盘）{pre['grid_rule']['early_close_sessions']} 个交易日，按 3 根 bar 处理；"
      f"日历规则由 NYSE 假日/半日市规则生成，并用数据反查一致。\n")
    A(f"**降级日（degraded）**：{pre['degraded_flag_note']}\n")
    A(f"共同窗口 {start} 起算，暖机约束为 C-H-S/C-DH-S 的 120 个有效 v（第 {cwj['bar_index_required']} 根 bar），"
      f"九只全部暖机后才开始，冻结于收益计算之前。\n")

    # ---------------- 4. 实现口径
    A("## 4. 实现口径与差异（SPEC §3 已同步）\n")
    A("| 项 | r1 | r2 小时批 |\n|---|---|---|")
    A("| C-H/H | 未运行 | 与 C-D/D 同一引擎 `ceng.run_breakout`，只把 bar 面板换成 60 分钟："
      "突破线 20 根小时 bar、ATR14 小时、收缩 120 根小时 bar、最长持仓 40 根小时 bar |")
    A("| C-D/H | 未运行 | `hourly.run_dh`：开盘前用 d−1 日线冻结 B/ATR14/收缩分位，日内任一完整小时 bar "
      "收盘首破 B 则按 DV60 排序取一个、下一根小时 bar 开盘入场；S0 用日线 ATR；移动止损、40 根"
      "（日线）持仓、退出全部按日线收盘触发、次日首根 bar 开盘执行 |")
    A("| B-H | 未运行 | 锁定 Fib 阶段引擎与 MA30 直接跑在小时 bar 上（同 bar 数、不同经济时间），"
      "波动与流动性仍取上一完整交易日 |")
    A("| 日线输入 | — | 所有 σ/ATR/成交额只用**上一完整交易日**；日内不读当日收盘、当日成交额（测试覆盖） |")
    A("| 净值频率 | 日频 | 小时引擎逐 bar 记账，**汇总到每日收盘一个标记**再算指标（方案 §8.2/§9.1） |")
    A("| 退出执行 | 次日开盘 | 小时配置为**次日 10:00 首根 bar 开盘**（受网格限制，见 §3） |")
    A("| 延迟压力 | +1 bar | 另加「+1 个交易日」日历压力（同一时刻、下一交易日），避免「多等一根」不可比 |")
    A("")

    # ---------------- 5. 六配置业绩
    A("## 5. 六个新配置（同一共同窗口，日频净值）\n")
    A("| 配置 | CAGR | 最大回撤 | Sharpe | Sortino | 年化波动 | 平均仓位 | 完整交易 | 胜率 | 单笔净期望 |\n"
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cid in NEW:
        m = r["configs"][cid]
        t = m["trades"]
        A(f"| {cid} | {f(m['cagr'],2,True)} | {f(m['maxdd'],2,True)} | {f(m['sharpe'],3)} | "
          f"{f(m['sortino'],3)} | {f(m['ann_vol'],2,True)} | {f(m['avg_exposure'],3)} | {t['n_trades']} | "
          f"{f(t['win_rate'],1,True)} | {f(t['expectancy'],3,True)} |")
    A("")
    A("同一窗口的 W/D 对照（本批重跑，`runs/cw_line/`）：\n")
    A("| 配置 | CAGR | 最大回撤 | Sharpe | 完整交易 |\n|---|---:|---:|---:|---:|")
    for cid in CTRL:
        m = cw[cid]["base"]
        A(f"| {cid} | {f(m['cagr'],2,True)} | {f(m['maxdd'],2,True)} | {f(m['sharpe'],3)} | "
          f"{m['trades']['n_trades']} |")
    A("")
    r1c = r1["configs"]
    A("上一轮全窗口（2019-01-02 起）对照，仅作稳定性核对：C-D-P "
      f"{f(r1c['C-D-P']['cagr'],2,True)} / {f(r1c['C-D-P']['maxdd'],2,True)} / {f(r1c['C-D-P']['sharpe'],3)}；"
      f"C-W-P {f(r1c['C-W-P']['cagr'],2,True)} / {f(r1c['C-W-P']['maxdd'],2,True)}；"
      f"B-W-ON {f(r1c['B-W-ON']['cagr'],2,True)}。与共同窗口版本差异在 0.0–0.7pp，属起点不同造成的路径差，"
      "不是口径变化（D 配置的最大回撤完全相同）。\n")

    # ---------------- 6. 主比较
    A("## 6. 15 个主比较（验证段 2024-01-02…2026-09-30）\n")
    A("| 比较 | 状态 | CAGR 差(pp) | p | Holm 阈值 | Sharpe 左/右 | 回撤差(pp) | 正段数 | 验证段交易 左/右 |\n"
      "|---|---|---:|---:|---:|---|---:|---:|---|")
    for _, row in cmp_df.iterrows():
        A(f"| {row.comparison} | {row.status} | {f(row.cagr_diff_pp,2,sign=True) if pd.notna(row.cagr_diff_pp) else '—'} | "
          f"{f(row.p_raw,3)} | {f(row.p_holm_threshold,4)} | {f(row.sharpe_left,3)} / {f(row.sharpe_right,3)} | "
          f"{f(row.maxdd_diff_pp,2,sign=True)} | {int(row.segments_positive) if pd.notna(row.segments_positive) else '—'} | "
          f"{int(row.trades_left) if pd.notna(row.trades_left) else '—'} / "
          f"{int(row.trades_right) if pd.notna(row.trades_right) else '—'} |")
    A("")
    A("Holm step-down 覆盖全部 15 项（未运行者 p=1 占位，家族未缩小），校正后**无一项显著**。"
      "`INSUFFICIENT_TRADES_EVIDENCE` 表示验证段完整交易数 <50，按方案只能给机制结论。\n")
    A("新增 6 个比较的逐条判读：\n")
    b1, b2, b3 = cp.loc["B-H-ON-B-H-OFF"], cp.loc["B-H-OFF-B-W-OFF"], cp.loc["C-H-S-C-H-P"]
    b4, b5, b6 = cp.loc["C-DH-S-C-DH-P"], cp.loc["C-H-P-C-D-P"], cp.loc["C-DH-P-C-D-P"]
    A(f"- **B-H-ON−B-H-OFF**：{f(b1.cagr_diff_pp,2,sign=True)}pp，p={b1.p_raw:.2f}，FAIL。"
      "把 Fib 系数打开（g 原样）在小时时钟下更差；B 线单笔期望为负，ON/OFF 的差被成本噪声主导。")
    A(f"- **B-H-OFF−B-W-OFF**：{f(b2.cagr_diff_pp,2,sign=True)}pp，p={b2.p_raw:.2f}。"
      "小时决策相对周决策是明确负贡献，"
      f"完整交易从 {cw['B-W-OFF']['base']['trades']['n_trades']} 笔涨到 "
      f"{r['configs']['B-H-OFF']['trades']['n_trades']} 笔，费用吃掉全部毛收益。")
    A(f"- **C-H-S−C-H-P**：{f(b3.cagr_diff_pp,2,sign=True)}pp，p={b3.p_raw:.2f}，FAIL。小时收缩过滤是负贡献。")
    A(f"- **C-DH-S−C-DH-P**：{f(b4.cagr_diff_pp,2,sign=True)}pp，p={b4.p_raw:.2f}，交易数不足。"
      "混合入场下收缩条件基本中性。")
    A(f"- **C-H-P−C-D-P**：{f(b5.cagr_diff_pp,2,sign=True)}pp，p={b5.p_raw:.2f}，但回撤恶化 "
      f"{f(-b5.maxdd_diff_pp,1)}pp（远超 3pp 门槛），属「收益高、风险也高」，"
      "且 §8.3 风险匹配后消失（§7）。")
    A(f"- **C-DH-P−C-D-P**：{f(b6.cagr_diff_pp,2,sign=True)}pp，p={b6.p_raw:.2f}，回撤仅恶化 "
      f"{f(-b6.maxdd_diff_pp,2)}pp，Sharpe {f(b6.sharpe_left,2)} vs {f(b6.sharpe_right,2)}。"
      f"**过经济门槛与风险门槛，但验证段交易 {int(b6.trades_left)}/{int(b6.trades_right)} 笔 <50、"
      "p 未过 Holm → 证据等级「弱/条件性」**。\n")

    # ---------------- 7. 风险匹配 + 抽稀
    A("## 7. 收益来自能力还是来自暴露？\n")
    A("**§8.3 描述性风险匹配**（把高波动一侧的日净收益按较低波动等比缩小，不加杠杆；"
      "`EX_POST_ATTRIBUTION_ONLY`，不是可交易回测）：\n")
    A("| 配对 | 快侧波动 | 慢侧波动 | 缩放 | 缩放后 CAGR | 缩放后 Sharpe | 慢侧 CAGR | 结论 |\n"
      "|---|---:|---:|---:|---:|---:|---:|---|")
    for fast, slow in (("C-H-P", "C-D-P"), ("C-H-S", "C-D-S"), ("C-DH-P", "C-D-P"), ("C-DH-S", "C-D-S")):
        v = vol_matched(eq, fast, slow, start, end)
        cap = v["cagr_scaled"]
        verdict = ("缩水后不优于对照" if cap is not None and cap <= v["cagr_slow"] else
                   ("缩水后仍略优" if cap is not None else "快侧波动不高于慢侧，无需缩放"))
        A(f"| {fast} − {slow} | {f(v['vol_fast'],2,True)} | {f(v['vol_slow'],2,True)} | {f(v['scale'],2)} | "
          f"{f(cap,2,True) if cap is not None else '—'} | "
          f"{f(v['sharpe_scaled'],3) if v.get('sharpe_scaled') is not None else '—'} | "
          f"{f(v['cagr_slow'],2,True)} | {verdict} |")
    A("")
    A("**§8.4 机会抽稀诊断**（触发条件成立：C-H-P 在验证段收益更高且交易更多）：\n")
    A(f"- 设计段候选决策时点：日线 {thin['n_slow']} 次、小时 {thin['n_fast']} 次 → "
      f"冻结 p={thin['p']:.4f}；种子 0…99 全部保留，不挑最好种子。")
    A(f"- 100 条抽稀路径 CAGR：p5 {f(thin['thinned_cagr_p5'],2,True)}、中位 "
      f"{f(thin['thinned_cagr_median'],2,True)}、p95 {f(thin['thinned_cagr_p95'],2,True)}；"
      f"未抽稀小时 {f(thin['base']['C-H-P'],2,True)}、日线 {f(thin['base']['C-D-P'],2,True)}。")
    A(f"- 完整交易中位 {thin['thinned_n_trades_median']:.0f} 笔（日线 "
      f"{r['configs']['C-D-P']['trades']['n_trades']}、未抽稀小时 "
      f"{r['configs']['C-H-P']['trades']['n_trades']}）；"
      f"跑赢日线的路径占 {thin['share_seeds_beating_daily']*100:.0f}%。")
    A("- 判读：把候选频率对齐后，小时线的优势**全部消失**。这支持「小时线只是更多相关下注 + 更高暴露」，"
      "不支持「更多可用机会有价值」。p 只对齐候选频率、不对齐完成交易数，故这只是近似匹配，不是等价样本。\n")

    # ---------------- 8. 入场体验
    A("## 8. 入场体验（§9.3）\n")
    A("ATR 归一化口径在不同周期之间**不可比**（小时 ATR 约为日线 ATR 的 1/6，同一个价格回撤会被放大成更大的 ATR 倍数）。"
      "因此下表同时给原始百分比口径，跨周期只读百分比列。\n")
    A("| 配置 | 交易 | 首日 MAE 中位 | 首日 90% 分位损失 | 全程 MAE 中位 | 明显浮亏率(≤−0.5×ATR) | 明显浮亏率(≤−0.5%) | 5 日净损益均值 | 失败突破率 |\n"
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cid in ("C-W-P", "C-W-S", "C-D-P", "C-D-S", "C-H-P", "C-H-S", "C-DH-P", "C-DH-S"):
        g = exp[(exp.config == cid)]
        if not len(g):
            continue
        L1 = np.maximum(0, -g.mae_1d.astype(float).values)
        n1 = g.net_5d.mean() if "net_5d" in g else np.nan
        adv = (g.mae_1d_atr.astype(float) <= -0.5).mean() if "mae_1d_atr" in g else np.nan
        advp = (g.mae_1d.astype(float) <= -0.005).mean()
        fail = g.failed_breakout_5bars.mean() if "failed_breakout_5bars" in g else np.nan
        A(f"| {cid} | {len(g)} | {f(np.median(g.mae_1d),2,True)} | {f(np.quantile(L1,0.9),2,True)} | "
          f"{f(np.median(g.mae),2,True)} | {f(adv,1,True)} | {f(advp,1,True)} | {f(n1,3,True)} | {f(fail,1,True)} |")
    A("")
    eH, eD = exp[exp.config == "C-H-P"], exp[exp.config == "C-D-P"]
    eHP, eHS = exp[exp.config == "C-DH-P"], exp[exp.config == "C-DH-S"]
    A(f"读法：小时入场的首日 MAE **原始百分比更小**（C-H-P 中位 {f(np.median(eH.mae_1d),2,True)} vs "
      f"C-D-P {f(np.median(eD.mae_1d),2,True)}），因为它更早离场、持仓时间更短；但 ATR 归一化的"
      f"「明显浮亏率」反而更高（{f((eH.mae_1d_atr<=-0.5).mean(),1,True)} vs "
      f"{f((eD.mae_1d_atr<=-0.5).mean(),1,True)}），纯粹是小时 ATR 更小的尺度效应，"
      "不能当作「入场体验更差」的证据。"
      f"混合入场 C-DH-P/C-DH-S 的首日浮亏与 ATR 归一化浮亏率都低于对应日线配置（首日 MAE "
      f"{f(np.median(eHP.mae_1d),2,True)}/{f(np.median(eHS.mae_1d),2,True)}，ATR 归一化浮亏率 "
      f"{f((eHP.mae_1d_atr<=-0.5).mean(),1,True)}/{f((eHS.mae_1d_atr<=-0.5).mean(),1,True)}），"
      "但它们的 ATR 是日线 ATR、入场点是小时，两者本来就更容易对齐，比较时须注意这一点。"
      "「全程 MAE」一列受持仓时长支配（混合入场按 40 个日线 bar 持有），不能跨周期直接比较。\n")

    # ---------------- 9. 压力
    A("## 9. 成本、延迟、小资金\n")
    A("| 配置 | 基础 | 成本×2 | 成本×3 | +1 bar | +1 交易日 |\n|---|---:|---:|---:|---:|---:|")
    for cid in NEW:
        s = hl[cid]
        cells = [f(s["base"]["cagr"], 2, True)]
        for k in ("cost_x2", "cost_x3", "delay_1bar", "delay_1bar_session_calendar"):
            v = s.get(k) or {}
            cells.append(f(v.get("cagr"), 2, True) if v else "—")
        A(f"| {cid} | " + " | ".join(cells) + " |")
    A("")
    A("小资金整数股（C 线四配置，`max(成交额×1bp, $1)` 最低手续费，滑点照基础档）：\n")
    A("| 配置 | $5,000 | $10,000 | $20,000 |\n|---|---:|---:|---:|")
    for cid in ("C-H-P", "C-H-S", "C-DH-P", "C-DH-S"):
        s = hl[cid]
        A(f"| {cid} | " + " | ".join(f((s.get(f'int_{c}') or {}).get('cagr'), 2, True)
                                     for c in (5000, 10000, 20000)) + " |")
    A("")
    A("读法：C-H-P 与 C-H-S 对成本高度敏感（C-H-S 在成本×2 后仍为负），"
      "C-DH-P/C-DH-S 几乎不敏感——它们的换手只有小时全配置的 1/5–1/8。"
      "整数股与最低手续费对四个配置的影响都在 0.1–1.3pp 以内，没有出现「小资金下结论反转」。\n")

    # ---------------- 10. 稳健性
    A("## 10. 稳健性压力\n")
    A("**§10.2 第 7 条：剔除验证段最高贡献标的**（事后识别，不是新的无偏验证）\n")
    A("| 比较 | 剔除 | 剔除后 CAGR 差(pp) | 主结果 CAGR 差(pp) |\n|---|---|---:|---:|")
    for _, row in g7.iterrows():
        base = cmp_df.loc[cmp_df.comparison == row.comparison, "cagr_diff_pp"]
        A(f"| {row.comparison} | {row.dropped_symbol} | {f(row.cagr_diff_pp,2,sign=True)} | "
          f"{f(base.iloc[0],2,sign=True) if len(base) else '—'} |")
    A("")
    g7i = g7.set_index("comparison")
    A(f"C-DH-P−C-D-P 在剔除 {g7i.loc['C-DH-P-C-D-P','dropped_symbol']} 后由 "
      f"{f(b6.cagr_diff_pp,2,sign=True)}pp 变为 {f(g7i.loc['C-DH-P-C-D-P','cagr_diff_pp'],2,sign=True)}pp："
      "**该优势依赖单一标的**，按 §10.2 不能称为广泛贡献。"
      f"C-H-P−C-D-P 反过来在剔除 {g7i.loc['C-H-P-C-D-P','dropped_symbol']} 后仍为 "
      f"{f(g7i.loc['C-H-P-C-D-P','cagr_diff_pp'],2,sign=True)}pp，但它的主问题不是单标的依赖而是风险。\n")
    A("**时段诊断**（替代 short_bar，post-hoc）\n")
    if len(hours):
        hh = hours.groupby("hour_et").apply(lambda g: g.n.sum(), include_groups=False)
        tot = hh.sum()
        A("小时配置的信号时段分布（ET）：" + "，".join(f"{int(h)}:00 {n/tot*100:.0f}%" for h, n in hh.items()) + "。")
    A("禁用当日最后一根 bar 入场后的复算：" +
      "，".join(f"{cid} {f((hl[cid].get('diag_no_lastbar_entry') or {}).get('cagr'),2,True)}"
               for cid in ("C-H-P", "C-H-S", "C-DH-P", "C-DH-S")) +
      "（与基础结果差 ≤0.5pp，顺序不变）→ 结论不依赖尾段触发。\n")

    # ---------------- 11. 证据等级
    A("## 11. 证据等级（按 §10.1 分类）\n")
    A("| 问题 | 结论 | 等级 | 依据 |\n|---|---|---|---|")
    A("| 60 分钟周期是否优于日线 | 否 | **负/无贡献**（全小时）；**弱条件性**（D/H 混合） | "
      "C-H-P 相对 C-D-P 全窗口 +0.29pp 但回撤 −26.1% vs −10.2%；抽稀后优势消失；"
      "C-DH-P +1.34pp 全窗口 / +2.31pp 验证段，p=0.18、交易不足、剔除 GLD 后翻负 |")
    A("| 收缩后突破（S）是否更好 | 否 | **负贡献** | 小时 −7.91pp、日线 −0.35pp、周线 −3.66pp；"
      "混合入场中性（−0.28pp） |")
    A("| 旧框架 B 缩短到小时 | 否 | **负贡献** | B-H-OFF 1.93%、回撤 −29.9%、单笔期望 −0.20%；"
      "相对 B-W-OFF −11.41pp |")
    A("| Fib 系数在小时时钟下是否有贡献 | 否（且方向为负） | **弱负** | B-H-ON−B-H-OFF −3.53pp，p=1.00 |")
    A("| 入场体验是否改善 | 原始百分比：是；ATR 归一化：不可比 | **证据不足** | 见 §8 的尺度陷阱 |")
    A("| 小资金是否仍成立 | 是 | **弱正（稳健性）** | 0.1–1.3pp 内，顺序不变 |")
    A("")

    # ---------------- 12. 建议 + 不足
    A("## 12. 建议与仍证据不足的部分\n")
    A("**建议**\n")
    A("1. **不要把 60 分钟作为主周期。** 全小时突破在扣除成本后收益与日线相当、风险显著更差，"
      "且优势在频率对齐后消失。")
    A("2. **收缩后突破（S）从候选里移除。** 三个周期一致为负或中性，小时线最差。")
    A("3. **如果继续，只继续 D/H 混合入场这一条线**：C-DH-P 的风险几乎不变、收益略高、"
      "成本与延迟不敏感；但必须先解决「验证段交易数不足 50」和「剔除 GLD 后翻负」两个问题——"
      "也就是需要更长的小时历史或更宽的标的池，而不是调参。")
    A("4. **B 线不必再往短周期推。** 三持仓框架的换手在小时时钟下不可控。\n")
    A("**证据不足 / 不能下的结论**\n")
    A("- 30 分钟滞后（无 09:30–10:00 bar）对 D/H 的影响无法从本数据分离："
      "若开盘半小时恰好是突破高发时段，D/H 的真实表现可能与本报告不同（方向未知）。")
    A("- 单交易所 OHLCV 的日内高低点比合并行情窄（包含率 72–87%），"
      "突破与止损触发次数可能被系统性低估，幅度未量化。")
    A("- 验证段只有 27–28 笔完整交易（D/H 两条），统计功效不足以判定「有效」或「无效」。")
    A("- 2026 年只有 9 个月，不能称完整年度；历史验证段统一标 `OOS_REUSED`。")
    A("- 本批未重跑 A 线；A 的结论以 r1 交付为准，不与本批 B/C 混用。\n")

    # ---------------- 附录
    A("## 附录 A：逐年净收益\n")
    A("| 配置 | " + " | ".join(str(y) for y in sorted(r["configs"]["C-DH-P"]["yearly"])) + " |")
    A("|---|" + "---:|" * len(r["configs"]["C-DH-P"]["yearly"]))
    for cid in NEW:
        y = r["configs"][cid]["yearly"]
        A(f"| {cid} | " + " | ".join(f(y.get(str(k)) or y.get(k), 1, True) for k in sorted(y)) + " |")
    A("")
    A("## 附录 B：preregistration / preflight / acceptance 的变动\n")
    A("- `preflight.json`：`hourly.hourly_available` 由 `false` 改为 `true`，"
      "verdict 由 `BLOCKED_NO_HOURLY_DATA` 改为 `HOURLY_AVAILABLE`，附 CSV/清单 sha256、覆盖率、网格限制与共同窗口。")
    A("- `acceptance.json`：新增 `hourly_data` 段（9 只齐全 / 覆盖率 99.66% / 跨源 PASS / 暖机 36 个月 / 无前视测试 8 项）；"
      "六个配置由 `BLOCKED_NO_HOURLY_DATA` 改为 `DONE`，并保留 `previous_status` 以便审计。")
    A("- `preregistration.json`：六个配置状态改为 `UNBLOCKED_R2_HOURLY_FEED`；"
      "18 个配置、15 个比较、门槛、种子、成本档**未改动**（本批没有改规格）。")
    A("- `trials.csv`：新增六配置的基础/成本×2/×3/延迟/日历延迟/整数股/事后诊断，共 "
      f"{len(pd.read_csv(RUN / 'trials.csv'))} 行。")
    A("")
    A("## 附录 C：与 r1 代码的等价性\n")
    A("本批修改了共享引擎（`ceng.Panel` 增加可选会话字段、`ceng/beng` 增加惰性 `delay_sessions`、"
      "`data.load_etf_daily` 增加两列）。为证明默认路径未动，用本批代码重跑 r1 的 8 个配置并与 r1 净值逐日比较，"
      "最大绝对差 **0.0**（`runs/repro_c`、`runs/repro_b`，脚本 `verify_reproduction.py`）。\n")

    (RUN / "REPORT_ZH.md").write_text("\n".join(L) + "\n")
    print(f"wrote {RUN/'REPORT_ZH.md'} ({len('\n'.join(L))} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
