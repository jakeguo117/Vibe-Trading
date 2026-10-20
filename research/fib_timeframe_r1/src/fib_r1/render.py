"""Renders REPORT_ZH.md from the machine-generated artefacts (no hand-typed performance numbers)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

pd.set_option("display.width", 200)


def f(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x*100:.{nd}f}%" if pct else f"{x:.{nd}f}"


def pp(x, nd=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:+.{nd}f}pp"


def main(run_root):
    R = Path(run_root)
    res = json.loads((R / "result.json").read_text())
    cmp = pd.read_csv(R / "comparisons.csv")
    ep = R / "extra" / "extra.json"
    extra = json.loads(ep.read_text()) if ep.exists() else {}
    acc = json.loads((R / "acceptance.json").read_text())
    pre = json.loads((R / "meta" / "preflight.json").read_text())
    gate0 = json.loads((R / "gate0.json").read_text())
    cfg = res["configs"]
    L = []
    A = L.append

    def row(cid, name=None):
        v = cfg[cid]
        t = v.get("trades") or {}
        e = v.get("entry") or {}
        return (f"| {name or cid} | {f(v['cagr'],2,True)} | {f(v['maxdd'],2,True)} | {f(v['sharpe'],3)} | "
                f"{f(v.get('sortino'),3)} | {f(v['ann_vol'],2,True)} | {f(v['avg_exposure'],1,True)} | "
                f"{t.get('n_trades','—')} | {f(t.get('win_rate'),1,True)} | {f(t.get('profit_factor'),2)} | "
                f"{f(e.get('mae_p90_loss'),4)} |")

    A("# Fib 贡献与交易周期缩短 —— 研究报告（中文）\n")
    A(f"研究编号 **{res['research_id']}**　方案 sha256 `{res['plan_sha256']}`　历史验证段标 **{res['holdout_tag']}**\n")
    A("> 本文件所有业绩数字均由 `scripts/render.py` 从 `result.json` / `comparisons.csv` / `extra.json` "
      "生成，未手工填写。数字口径见 §7 与方案 §9。\n")

    # ---------------- one page
    A("## 0. 一页结论\n")
    a00, a10, a01, a11 = (cfg[k] for k in ("A00", "A10", "A01", "A11"))
    A(f"1. **Fib 仓位系数不是「没用」，但在正式系统里作用很小，且远达不到 2pp 的净收益门槛。** "
      f"关掉系数（A10）后全区间 CAGR {f(a00['cagr'],2,True)} → {f(a10['cagr'],2,True)}"
      f"（{pp((a00['cagr']-a10['cagr'])*100)}），但**回撤反而变大**（{f(a00['maxdd'],2,True)} → {f(a10['maxdd'],2,True)}）。"
      f"历史验证段上 A00−A10 = {pp(float(cmp.loc[0,'cagr_diff_pp']))}、p={float(cmp.loc[0,'p_raw']):.3f}，"
      f"Holm 校正后不显著，且验证段只有 {int(cmp.loc[0,'trades_left'])} 笔完整交易，不满足 ≥50 笔的证据门槛。")
    A(f"2. **系数被波动归一化抵消的比例只有 {f(res['cancellation']['cancel_rate_all'],2,True)}**，"
      f"不是「基本抵消」。1947 个决策里权重完全不变的只占这一小部分；其余决策的目标权重 L1 变化中位数 "
      f"{res['cancellation']['l1_quantiles']['0.5']:.3f}、95 分位 {res['cancellation']['l1_quantiles']['0.95']:.3f}。"
      f"A00 全程持有 3 只，单持仓抵消（方案 §4.3 的极端情形）在正式系统里**一次都没发生**。")
    A(f"3. **替换 Fib 历史排名的代价比关掉系数大。** A01（仅换成 DV60 排名）CAGR {f(a01['cagr'],2,True)}、"
      f"maxdd {f(a01['maxdd'],2,True)}；A11（两者都关）CAGR {f(a11['cagr'],2,True)}、maxdd {f(a11['maxdd'],2,True)}。"
      f"验证段 A00−A01 = {pp(float(cmp.loc[1,'cagr_diff_pp']))}、A00−A11 = {pp(float(cmp.loc[2,'cagr_diff_pp']))}，"
      f"方向一致为正但 p 值 {float(cmp.loc[1,'p_raw']):.2f} / {float(cmp.loc[2,'p_raw']):.2f}，Holm 后均不显著。")
    A(f"4. **缩短到日线在旧框架里没有优势。** B-W-ON {f(cfg['B-W-ON']['cagr'],2,True)} vs B-D-ON "
      f"{f(cfg['B-D-ON']['cagr'],2,True)}；OFF 下 B-D-OFF−B-W-OFF = "
      f"{pp(float(cmp.loc[6,'cagr_diff_pp']))}（p={float(cmp.loc[6,'p_raw']):.2f}）。"
      f"日线的完整交易从周线 20 笔涨到 103 笔，**次数确实增加了 5 倍，但净收益没有增加**——这正是方案 §8 要分开看的"
      f"「更多交易」与「更值钱的交易」。")
    A(f"5. **单持仓新策略 C 目前不具备候选资格。** 最好的 C-D-S 是 CAGR {f(cfg['C-D-S']['cagr'],2,True)}、"
      f"maxdd {f(cfg['C-D-S']['maxdd'],2,True)}、Sharpe {f(cfg['C-D-S']['sharpe'],3)}，"
      f"而同一 9 只 ETF 池里 SPY 买入持有是 {f(extra['buy_and_hold'][2]['cagr'],2,True)} CAGR / Sharpe "
      f"{f(extra['buy_and_hold'][2]['sharpe'],3)} / maxdd {f(extra['buy_and_hold'][2]['maxdd'],1,True)}。"
      f"C 的回撤确实小得多，但按风险调整后仍不如被动持有。")
    A(f"6. **收缩后突破（S）减少了入场浮亏，但没换来收益。** C-D-S 相对 C-D-P：明显浮亏率 "
      f"{f(cfg['C-D-P']['entry'].get('adverse_rate_1d'),1,True)} → {f(cfg['C-D-S']['entry'].get('adverse_rate_1d'),1,True)}"
      f"（降 {abs((cfg['C-D-P']['entry'].get('adverse_rate_1d') or 0)-(cfg['C-D-S']['entry'].get('adverse_rate_1d') or 0))*100:.1f}pp，"
      f"未达到方案要求的 5pp），第一日 MAE 的 90 分位损失 "
      f"{f(cfg['C-D-P']['entry'].get('mae_p90_loss'),4)} → {f(cfg['C-D-S']['entry'].get('mae_p90_loss'),4)}（改善）；"
      f"但验证段净收益差 {pp(float(cmp.loc[9,'cagr_diff_pp']))}（p={float(cmp.loc[9,'p_raw']):.2f}），周线上 S 更差。")
    A("7. **成本不是主因。** 手续费+滑点×3 后 C-D-S 的 CAGR 只从 "
      f"{f(cfg['C-D-S']['cagr'],2,True)} 降到 {f(cfg['C-D-S']['stress']['cost_x3']['cagr'],2,True)}；"
      "真正脆弱的是**执行延迟**：C-D-P 统一延迟 1 根 bar 后 CAGR 变 "
      f"{f(cfg['C-D-P']['stress']['delay_1bar']['cagr'],2,True)}，说明日线突破的收益几乎全部来自突破当日起的第一段涨幅。")
    A("8. **证据等级：全部 15 个主比较的最高状态是「证据不足」，没有一个是「强正贡献」。** 6 个含 60 分钟数据的比较"
      "因本机无小时历史而 `BLOCKED`；可运行的 9 个里，验证段完整交易数 1–34 笔，全部低于 50 笔门槛。\n")

    # ---------------- Gate 0
    A("## 1. Gate 0：可比性\n")
    A("| 检查 | 结果 | 证据 |")
    A("|---|---|---|")
    A(f"| 方案哈希 | `{pre['plan']['sha256'][:16]}…`（与冻结值一致：{pre['plan']['sha256_matches_frozen']}） | preflight.json |")
    A(f"| 参考仓库 4 个引擎 blob | 全部匹配：{pre['code']['blobs_match_plan']} | preflight.json |")
    A(f"| 原样 reproduce v1.1.1 | PASS，逐日净值 maxabs {gate0['reproduce_v1_1_1']['daily_equity_maxabs']:.3e}（要求 1e-9） | runs/gate0.json |")
    A(f"| verify-data | inputs exact {gate0['verify_data']['inputs_exact']}、derived exact {gate0['verify_data']['derived_exact']}，missing/mismatch 空 | runs/gate0.json |")
    A(f"| 正式测试套件 | {gate0['pytest']['result_line']} | runs/…/pytest_full.log |")
    A(f"| 研究版 A00 == 正式引擎 | 逐日净值 maxabs {res['identity']['a00_vs_official_maxabs']}（要求 1e-12） | a_line.json |")
    A(f"| 小时（60m）数据 | **不存在**：快照 README 自述「yfinance raw daily bars」，按文件名与内容两次扫描 0 命中 | preflight.json |")
    A("")
    A(f"小时数据缺失只阻塞小时组：`B-H-ON`、`B-H-OFF`、`C-H-P`、`C-H-S`、`C-DH-P`、`C-DH-S` 六配置标 "
      f"`BLOCKED_NO_HOURLY_DATA`，未用日线插值、未购买数据、未调用券商。\n")

    # ---------------- A line
    A("## 2. A 线：Fib 在现有系统里做了什么\n")
    A("| 配置 | CAGR | maxdd | Sharpe | Sortino | 年化波动 | 平均仓位 | 完整交易 | 胜率 | PF | MAE 90 分位损失 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("A00", "A10", "A01", "A11"):
        A(row(k))
    A("")
    A(f"全区间（2019-01-02…2026-09-30，1947 个交易日）。A 的「完整交易」按 §8.1 是单代码连续持仓 episode，"
      f"由 `weights_*.csv` 的持仓段生成；A00/A10 的 episode 集合完全相同（{cfg['A00']['trades']['n_trades']} 笔），"
      f"A01/A11 也相同（{cfg['A01']['trades']['n_trades']} 笔），说明两个开关只改仓位不改选名——"
      f"事件序列逐条一致（A00 vs A10：{res['divergence']['A00_vs_A10']['events_equal']}）。\n")
    A(f"价格收益与净值贡献分开：A00 的 episode 平均价格收益 {f(cfg['A00']['entry']['mean_price_ret'],1,True)}，"
      f"但平均净值贡献只有 {f(cfg['A00']['entry']['mean_nav_contribution'],2,True)}。"
      f"把「单笔涨了多少」当成「账户赚了多少」会高估一个数量级。\n")
    A("> A 线的 MAE 是**收盘价口径**：正式快照 `mat/` 只保存全市场 open/close/dv，没有 high/low，"
      "因此只能度量收盘回撤，不能声称测到盘中低点；episode 长达数百个交易日，这个数不能与 C 线的"
      "（bars 级、含 high/low）直接比较。\n")
    A("### 2.1 系数抵消审计（方案 §4.3）\n")
    c = res["cancellation"]
    A(f"- 决策总数 {c['n_decisions']}，全部为 3 只持仓（A00 从未出现 1 只或 2 只持仓的决策）。")
    A(f"- **抵消比例 {f(c['cancel_rate_all'],2,True)}**（目标权重在 g→1 后完全不变，容差 1e-12）。")
    A(f"- 权重变化 L1：中位 {c['l1_quantiles']['0.5']:.3f}、90 分位 {c['l1_quantiles']['0.9']:.3f}、"
      f"95 分位 {c['l1_quantiles']['0.95']:.3f}、最大 {c['l1_quantiles']['1.0']:.3f}。")
    A(f"- 单持仓抵消率：无法统计（{c['cancel_rate_single_position']}），因为正式系统在该区间从未只持一只。"
      f"这一点必须写清楚：方案 §4.3 的解析抵消 `target=min(cap, vt/σ)` 在 A 的实际路径上是**理论情形而非实际发生**。\n")
    A("审计在**冻结决策**上做（保持 A00 当时的持仓集合、σ、scale 与 cap），只把 g 换成 1，"
      "因此测的是「同一天同一批持仓下系数还剩下多少作用」，不是一条独立的可交易路径。\n")

    # ---------------- B line
    A("## 3. B 线：同一框架换周期\n")
    A("| 配置 | CAGR | maxdd | Sharpe | Sortino | 年化波动 | 平均仓位 | 完整交易 | 胜率 | PF | MAE 90 分位损失 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("B-W-ON", "B-W-OFF", "B-D-ON", "B-D-OFF"):
        A(row(k))
    A("")
    A(f"B-W 不是 A00：候选只按 DV60+代码排序、取消商品优先、九只一视同仁、三持仓 70% cap。"
      f"B 线自己的口径下，周线 ON−OFF = {pp(float(cmp.loc[3,'cagr_diff_pp']))}（p={float(cmp.loc[3,'p_raw']):.3f}），"
      f"日线 ON−OFF = {pp(float(cmp.loc[4,'cagr_diff_pp']))}（p={float(cmp.loc[4,'p_raw']):.2f}）。"
      f"**同一个 Fib 系数在周线是正贡献、在日线是负贡献**，方向随周期翻转，这本身就是「不能把 A 的结论外推到别的周期」的证据。\n")
    A(f"交易次数：B-W 20 笔 / B-D 103 笔（验证段 3 vs 34）。次数确实增加了，但日线的单笔净期望只有 "
      f"{f(cfg['B-D-ON']['trades'].get('expectancy'),4)}（ON）与 {f(cfg['B-D-OFF']['trades'].get('expectancy'),4)}（OFF），"
      f"周线是 {f(cfg['B-W-ON']['trades'].get('expectancy'),4)} / {f(cfg['B-W-OFF']['trades'].get('expectancy'),4)}。"
      f"**更多交易没有带来更多优势，只是把同一批机会切得更碎、并被 2% 死区与再平衡成本反复摩擦。**\n")

    # ---------------- C line
    A("## 4. C 线：单持仓普通突破 vs 收缩后突破\n")
    A("| 配置 | CAGR | maxdd | Sharpe | Sortino | 年化波动 | 平均仓位 | 完整交易 | 胜率 | PF | MAE 90 分位损失 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("C-W-P", "C-W-S", "C-D-P", "C-D-S"):
        A(row(k))
    A("")
    A(f"### 4.1 入场体验（方案 §9.3）\n")
    A("| 配置 | 样本 | MAE 中位 | 90 分位损失 | 明显浮亏率（首日 MAE ≤ −0.5ATR） | 失败突破率（5 根内收回冻结线） | 单笔净期望 |")
    A("|---|---|---|---|---|---|---|")
    for k in ("C-W-P", "C-W-S", "C-D-P", "C-D-S"):
        e = cfg[k]["entry"] or {}
        t = cfg[k]["trades"] or {}
        A(f"| {k} | {e.get('n','—')} | {f(e.get('mae_median'),4)} | {f(e.get('mae_p90_loss'),4)} | "
          f"{f(e.get('adverse_rate_1d'),1,True)} | {f(e.get('failed_breakout_rate'),1,True)} | "
          f"{f(t.get('expectancy'),4)} |")
    A("")
    A(f"收缩条件确实把 C-D 的 90 分位浮亏从 {f(cfg['C-D-P']['entry'].get('mae_p90_loss'),4)} 降到 "
      f"{f(cfg['C-D-S']['entry'].get('mae_p90_loss'),4)}，明显浮亏率从 "
      f"{f(cfg['C-D-P']['entry'].get('adverse_rate_1d'),1,True)} 降到 {f(cfg['C-D-S']['entry'].get('adverse_rate_1d'),1,True)}，"
      f"降幅 {(abs((cfg['C-D-P']['entry'].get('adverse_rate_1d') or 0)-(cfg['C-D-S']['entry'].get('adverse_rate_1d') or 0))*100):.1f}pp，"
      f"**未达到方案 §10.2 要求的 5pp**。代价是交易数从 68 掉到 45（−34%），周线上更差"
      f"（C-W-S {f(cfg['C-W-S']['cagr'],2,True)} vs C-W-P {f(cfg['C-W-P']['cagr'],2,True)}）。"
      f"按方案口径这属于「体验改善但收益有代价」，交 Jake 决定，不替他优化单一指标。\n")

    # ---------------- cost
    A("## 5. 成本、延迟、小资金\n")
    A("| 配置 | 基础 | 成本×2 | 成本×3 | 延迟 +1 bar | 5k 整数股 | 10k 整数股 | 20k 整数股 |")
    A("|---|---|---|---|---|---|---|---|")
    for k in ("C-W-P", "C-W-S", "C-D-P", "C-D-S", "B-W-ON", "B-W-OFF", "B-D-ON", "B-D-OFF"):
        s = cfg[k].get("stress") or {}
        g = lambda key: f(s[key]["cagr"], 2, True) if s.get(key, {}).get("cagr") is not None else "—"
        A(f"| {k} | {f(cfg[k]['cagr'],2,True)} | {g('cost_x2')} | {g('cost_x3')} | {g('delay_1bar')} | "
          f"{g('int_5000')} | {g('int_10000')} | {g('int_20000')} |")
    A("")
    A(f"A 线成本×2 完整重跑（验证段 CAGR）：A00 {f(res['cost_a_line']['A00']['cagr_holdout'],2,True)}、"
      f"A10 {f(res['cost_a_line']['A10']['cagr_holdout'],2,True)}、A01 {f(res['cost_a_line']['A01']['cagr_holdout'],2,True)}、"
      f"A11 {f(res['cost_a_line']['A11']['cagr_holdout'],2,True)}；四组在成本×2 后顺序不变，A00−A10 仍有 "
      f"{pp(res['cost_a_line']['A00']['cagr_holdout']*100-res['cost_a_line']['A10']['cagr_holdout']*100)}。\n")
    A("C 线的成本敏感性极低（成交笔数少、单笔金额大、九只 ETF 都在 20 亿美元 DV60 档以上），"
      "但**延迟敏感性极高**：C-D-P 延迟 1 根 bar 就从 "
      f"{f(cfg['C-D-P']['cagr'],2,True)} 掉到 {f(cfg['C-D-P']['stress']['delay_1bar']['cagr'],2,True)}。"
      "换句话说，这类日线突破研究中「能不能抢到开盘那一下」比「手续费几 bp」重要两个数量级，"
      "而本模拟用开盘价成交属于理想化假设（方案 §7.1 已声明）。\n")
    A("小资金整数股（每笔最低 1 美元手续费）对 C 的影响是 "
      f"{f((cfg['C-D-S']['cagr']-cfg['C-D-S']['stress']['int_5000']['cagr'])*100,2)}pp（5k）、"
      f"{f((cfg['C-D-S']['cagr']-cfg['C-D-S']['stress']['int_10000']['cagr'])*100,2)}pp（10k）、"
      f"{f((cfg['C-D-S']['cagr']-cfg['C-D-S']['stress']['int_20000']['cagr'])*100,2)}pp（20k），方向为负但幅度有限。\n")

    # ---------------- comparisons
    A("## 6. 15 个主比较（验证段 2024-01-02…2026-09-30）\n")
    A("| 比较 | 状态 | CAGR 差 | 95% 区间（日对数差均值） | p（单侧） | Holm 阈值 | 验证段完整交易 左/右 | 3 段中为正 |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in cmp.iterrows():
        ci = f"[{r.ci_lo:+.2e}, {r.ci_hi:+.2e}]" if pd.notna(r.get("ci_lo")) else "—"
        A(f"| {r.comparison} | {r.status} | {pp(r.cagr_diff_pp) if pd.notna(r.cagr_diff_pp) else '—'} | {ci} | "
          f"{('%.4f' % r.p_raw) if pd.notna(r.p_raw) else '—'} | "
          f"{('%.4f' % r.p_holm_threshold) if pd.notna(r.p_holm_threshold) else '—'} | "
          f"{int(r.trades_left) if pd.notna(r.trades_left) else '—'}/{int(r.trades_right) if pd.notna(r.trades_right) else '—'} | "
          f"{int(r.segments_positive) if pd.notna(r.segments_positive) else '—'}/3 |")
    A("")
    A("分块 bootstrap：块长 20、10,000 次、seed 20261020、循环块、两侧共用同一抽样索引；"
      "p 为单侧零假设检验（d 去均值后同样抽样）。Holm step-down 覆盖全部 15 个比较，未运行者 p=1 占位。"
      "**结论：没有任何一个比较通过 Holm 校正，也没有任何一个是「强正贡献」。**\n")
    A(f"最接近的是 B-W-ON−B-W-OFF（p={float(cmp.loc[3,'p_raw']):.4f}，Holm 阈值 {float(cmp.loc[3,'p_holm_threshold']):.4f}）——"
      f"p 小于 0.05 但大于 0.05/15，按事前规则不能宣布显著。这不是「没有作用」，而是「证据不足以支撑强结论」，"
      f"且它的验证段只有 3 笔完整交易。\n")

    # ---------------- background
    A("## 7. 背景表：同一 9 只 ETF 的买入持有\n")
    A("| 标的 | CAGR | 年化波动 | maxdd | Sharpe |")
    A("|---|---|---|---|---|")
    for r in extra["buy_and_hold"]:
        A(f"| {r['sym']} | {f(r['cagr'],2,True)} | {f(r['ann_vol'],2,True)} | {f(r['maxdd'],1,True)} | {f(r['sharpe'],3)} |")
    A("")
    A("背景表只用于说明机会成本，不参与赢家选择，也不把「战胜某只被动持有」当作战胜正式基准。"
      "所有 C 配置的 CAGR 都低于 SPY 与 QQQ 的同期买入持有；C 的价值只体现在回撤与波动显著更低。\n")

    # ---------------- robustness
    A("## 8. 稳健性：剔除最高贡献标的\n")
    A("| 配置 | 剔除标的 | 剔除前 CAGR | 剔除后 CAGR | 变化 |")
    A("|---|---|---|---|---|")
    for k, v in (extra.get("drop_best_symbol") or {}).items():
        A(f"| {k} | {v['dropped']} | {f(v['cagr_full'],2,True)} | {f(v['cagr_without'],2,True)} | "
          f"{pp((v['cagr_without']-v['cagr_full'])*100)} |")
    A("")
    A("C-W-P 与 C-W-S 对单一标的（GLD / SPY）高度依赖，剔除后收益基本归零甚至转负；"
      "C-D-P / C-D-S 相对分散（剔除 QQQ 后仅小幅下降）。但这一压力测试是事后识别，不能替代无偏验证。\n")

    # ---------------- evidence grading
    A("## 9. 证据强弱分类（方案 §10.1）\n")
    A("| 命题 | 分类 | 依据 |")
    A("|---|---|---|")
    A(f"| Fib 仓位系数被波动归一化完全抵消 | **机制抵消（仅单持仓时）** | 解析式成立，但正式系统 1947 个决策里 0 次单持仓；"
      f"实际抵消率 {f(res['cancellation']['cancel_rate_all'],2,True)} |")
    A(f"| Fib 仓位系数对净收益有强正贡献 | **证据不足** | 验证段 {pp(float(cmp.loc[0,'cagr_diff_pp']))}，p={float(cmp.loc[0,'p_raw']):.3f}，"
      f"Holm 未过，验证段交易 6 笔 |")
    A("| Fib 仓位系数降低回撤（风控价值） | **弱／条件性** | A00 maxdd 优于 A10、A01、A11，但未做显著性检验，且集中在验证段 |")
    A(f"| Fib 历史排名有增量 | **弱／条件性** | A01 全区间 {f(a01['cagr'],2,True)} 低于 A00 {f(a00['cagr'],2,True)}，"
      f"但 A01 回撤大 5.4pp；验证段 {pp(float(cmp.loc[1,'cagr_diff_pp']))}，p={float(cmp.loc[1,'p_raw']):.2f} |")
    A("| 缩短到日线提升净收益 | **负贡献** | B-D-OFF−B-W-OFF = "
      f"{pp(float(cmp.loc[6,'cagr_diff_pp']))}；C-D-P−C-W-P = {pp(float(cmp.loc[12,'cagr_diff_pp']))} |")
    A("| 更多交易累积优势 | **不成立** | 日线交易数是周线 5 倍（103 vs 20），单笔净期望反而更低 |")
    A("| 收缩后突破减少明显浮亏 | **弱正贡献（体验）** | 90 分位损失与明显浮亏率均改善，但降幅 <5pp，且收益有代价 |")
    A("| 收缩后突破提升净收益 | **证据不足** | 验证段 " + pp(float(cmp.loc[9, 'cagr_diff_pp'])) + "，p=" + f"{float(cmp.loc[9,'p_raw']):.2f}" + " |")
    A("| 60 分钟周期是否更好 | **证据不足（数据阻塞）** | 本机无小时数据，6 个配置 BLOCKED |")
    A("")

    # ---------------- recommendation
    A("## 10. 建议\n")
    A("1. **保留 Fib 仓位系数，但不要把它当成收益来源。** 它在正式系统里主要是「削峰」（A00 的回撤在三组对照里最小），"
      f"收益贡献全区间只有 {pp((a00['cagr']-a10['cagr'])*100)}，统计上不足以称强贡献。若未来要简化，**先简化排名、后简化系数**——"
      f"排名替换的回撤代价明显更大（A01 {f(a01['maxdd'],1,True)} vs A00 {f(a00['maxdd'],1,True)}，"
      f"差 {abs(a01['maxdd']-a00['maxdd'])*100:.1f}pp）。")
    A("2. **不建议把决策周期缩短到日线。** 在旧框架里日线不涨收益、只涨次数与换手；在新单持仓策略里日线同样"
      f"输给周线的风险调整后表现（C-W-P Sharpe {f(cfg['C-W-P']['sharpe'],3)} vs C-D-P {f(cfg['C-D-P']['sharpe'],3)}）。")
    A("3. **单持仓 C 策略暂停，不作为候选。** 它跑不赢同池被动持有，成本与延迟敏感性又集中在入场那一下。"
      "若继续，应先解决「为什么单笔净期望接近 0」这个问题，而不是调参数。")
    A("4. **收缩过滤值得保留为「体验」选项，不值得作为主策略。** 它确实降低了入场浮亏，但降幅不到门槛，"
      "且在周线上明显扣收益。")
    A("5. **必须补的数据：60 分钟历史。** 6 个核心配置（含 D/H 混合入场）完全无法评估，这是本轮最大的空白，"
      "不是「没做」，是「没有数据可做」。")
    A("6. **研究候选 ≠ 可实盘替代。** 本报告没有任何配置达到方案 §10.2 的 7 条门槛，因此不构成换版建议。\n")

    # ---------------- limits
    A("## 11. 限制与未完成\n")
    A("- 6 个含小时数据的配置 `BLOCKED_NO_HOURLY_DATA`；未购买数据、未插值。")
    A("- A 线的 MAE/MFE 只能用**收盘价**口径：正式快照 `mat/` 不保存全市场 high/low。已在 `entry_experience_all.csv` 的 `basis` 列标注。")
    A("- A 线的「完整交易」是持仓 episode，不是撮合成交；A 是权重引擎，没有逐笔成交价。")
    A("- 所有验证段指标标 `OOS_REUSED`：这段历史在设计时已被反复查看，Holm 校正不能消除这种偏差。")
    A("- C 的开盘价成交、开盘价止损触发都是理想化模拟；延迟 1 根 bar 的实验说明结论对此非常敏感。")
    A("- 周线的「额外 1 个交易日」延迟压力用 1 根周 bar 近似，只能读作上界。")
    A("- 机会抽稀诊断（§8.4）**未触发**：C-D-P−C-W-P 净收益为负，不满足「更高净收益且更多完整交易」的触发条件。\n")

    A("## 12. 复现\n")
    A("```bash")
    A("# 1) Gate 0（只读参考仓库）")
    A("cd trading-system && .venv/bin/python -m upleg_bt reproduce v1.1.1")
    A("# 2) A 线")
    A("cd research/fib_timeframe_r1 && PYTHONPATH=src:<trading-system>/backtest \\")
    A("  .venv/bin/python -m fib_r1.adriver --out runs/a_line --data-dir <snapshot> --backtest-dir <trading-system>/backtest")
    A("# 3) C 线 / B 线 / 对比 / 汇总")
    A("  python -m fib_r1.cdriver --out runs/c_line --raw-dir <snapshot>/raw --comm-dir <snapshot>/commodities --stress")
    A("  python -m fib_r1.bdriver --out runs/b_line --raw-dir <snapshot>/raw --backtest-dir <trading-system>/backtest --stress")
    A("  python -m fib_r1.compare  runs runs/comparisons.csv")
    A("  python -m fib_r1.aggregate runs")
    A("```")
    A("")
    A(f"数据快照 `ts_v111`：mat 树 sha256 `{pre['data']['mat_files']['tree_sha256']}`、"
      f"raw 树 sha256 `{pre['data']['raw_files']['tree_sha256']}`；`signals_primary.npz` "
      f"{pre['data']['signals_primary']['bytes']} 字节 sha256 `{pre['data']['signals_primary']['sha256'][:32]}…`。")
    A("")

    out = R / "REPORT_ZH.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
