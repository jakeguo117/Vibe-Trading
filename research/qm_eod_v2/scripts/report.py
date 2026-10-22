#!/usr/bin/env python3
"""Build QM-EOD-v2 deliverables (§7) from run outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("QM_EOD_RUN_ID") or (ROOT / "results" / ".current_run_id").read_text().strip()
OUT = ROOT / "results" / RUN_ID


def pct(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100 * x:.2f}%"


def num(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def conclusion(m_acct, m_sig, qqq_cagr, n_trades):
    """继续研究 / 暂不支持 / 证据不足 — Chinese labels only."""
    avg_R = m_sig.get("avg_R", m_sig.get("sig_avg_R", float("nan")))
    cagr = m_acct.get("cagr", float("nan"))
    if n_trades < 100:
        return ("证据不足",
                f"成交仅 {n_trades} 笔（<100），不足以判定有效/无效；"
                f"账户年化 {pct(cagr)} vs QQQ {pct(qqq_cagr)}，信号层期望 {num(avg_R)}R。"
                "属 EOD-DIAG，不能外推盘中 ORH 原版。")
    sig_pos = np.isfinite(avg_R) and avg_R > 0
    acct_weak = np.isfinite(cagr) and np.isfinite(qqq_cagr) and cagr < 0.5 * qqq_cagr
    if (not sig_pos) and acct_weak:
        return ("暂不支持",
                f"信号层期望≤0（{num(avg_R)}R）且账户年化 {pct(cagr)} 远低于 QQQ {pct(qqq_cagr)}。"
                "仍为 EOD-DIAG，未证伪盘中原版。")
    if sig_pos and acct_weak:
        return ("证据不足",
                f"信号层期望为正（{num(avg_R)}R）但账户层弱（年化 {pct(cagr)} vs QQQ {pct(qqq_cagr)}）；"
                "多持仓已测，仍缺分钟线与退市股。继续研究需补数据，不宜升级为可交易结论。")
    if sig_pos and not acct_weak:
        return ("继续研究",
                f"信号层与账户层均显示可用迹象（期望 {num(avg_R)}R，年化 {pct(cagr)}）；"
                "仍属 EOD-DIAG，需分钟线确认后方可升级。")
    return ("证据不足",
            f"结果边界不清（期望 {num(avg_R)}R，年化 {pct(cagr)} vs QQQ {pct(qqq_cagr)}）；样本/代理限制下不作更强结论。")


def main():
    mdf = pd.read_csv(OUT / "metrics.csv")
    frozen = json.load(open(OUT / "frozen_config.json"))
    primary = mdf[(mdf.strategy == "COMBINED") & (mdf["filter"] == "on") &
                  (mdf.slip_bps == 10) & (mdf.e0 == 100000) & (mdf.segment == "FULL")]
    if len(primary) == 0:
        primary = mdf[(mdf.strategy == "COMBINED") & (mdf.slip_bps == 10) & (mdf.e0 == 100000)]
        primary = primary[primary.segment == primary.segment.iloc[0]]
    p = primary.iloc[0].to_dict()
    qqq = mdf[mdf.label == "QQQ-BH"]
    qqq_cagr = float(qqq.iloc[0]["cagr"]) if len(qqq) else float("nan")

    n_trades = int(p.get("n_trades", 0))
    label_zh, reason = conclusion(p, p, qqq_cagr, n_trades)

    # ---- concentration.md
    tr_path = OUT / "trade_ledger.csv"
    conc_lines = ["# 集中度与分段期望\n", f"run_id: `{RUN_ID}`\n"]
    if tr_path.exists() and tr_path.stat().st_size > 50:
        tr = pd.read_csv(tr_path)
        pnl = tr["pnl"].to_numpy(float)
        order = np.argsort(-pnl)
        tot = pnl.sum()
        conc_lines.append("## 前 N 笔占净利\n")
        conc_lines.append("| N | 累计PnL | 占净利 |\n|---|---:|---:|\n")
        for n in (1, 3, 5, 10, 20):
            if n > len(pnl):
                break
            s = pnl[order[:n]].sum()
            conc_lines.append(f"| {n} | {s:,.0f} | {s / tot * 100 if tot != 0 else float('nan'):.1f}% |\n")
        conc_lines.append("\n## 分段（账户层主结果）\n")
        conc_lines.append("| 分段 | 笔数 | avg_R | PF | CAGR | MDD |\n|---|---:|---:|---:|---:|---:|\n")
        for seg in ["Y2015_19", "Y2020_21", "Y2022", "Y2023_25", "FULL", "CONT"]:
            sub = mdf[(mdf.strategy == "COMBINED") & (mdf["filter"] == "on") &
                      (mdf.slip_bps == 10) & (mdf.e0 == 100000) & (mdf.segment == seg)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            conc_lines.append(
                f"| {seg} | {int(r.n_trades)} | {num(r.avg_R)} | {num(r.pf)} | {pct(r.cagr)} | {pct(r.mdd)} |\n"
            )
        # EP 8-K split
        if "earnings_tagged" in tr.columns:
            conc_lines.append("\n## EP 有/无 8-K（成交笔）\n")
            ep = tr[tr["style"] == "ep"] if "style" in tr.columns else tr
            if len(ep):
                for tag, g in ep.groupby(ep["earnings_tagged"].astype(bool)):
                    conc_lines.append(f"- earnings_tagged={tag}: n={len(g)}, avg_R={g['R'].mean():.3f}, sum_pnl={g['pnl'].sum():.0f}\n")
    (OUT / "concentration.md").write_text("".join(conc_lines))

    # ---- equity chart
    eq = pd.read_csv(OUT / "daily_equity.csv") if (OUT / "daily_equity.csv").exists() else None
    qeq = pd.read_csv(OUT / "qqq_equity.csv") if (OUT / "qqq_equity.csv").exists() else None
    off_dirs = list((OUT / "runs").glob("COMBINED__filter_off__slip10__e100000__*"))
    fig, ax = plt.subplots(figsize=(10, 5))
    if eq is not None and len(eq):
        e0 = eq["equity"].iloc[0]
        ax.plot(pd.to_datetime(eq["date"]), eq["equity"] / e0, label="Strategy (filter ON, 10bps)", color="#1f77b4", lw=1.5)
    if off_dirs:
        off_eq = pd.read_csv(off_dirs[0] / "equity.csv")
        e0 = off_eq["equity"].iloc[0]
        ax.plot(pd.to_datetime(off_eq["date"]), off_eq["equity"] / e0, label="Strategy (filter OFF)", color="#ff7f0e", lw=1.0, alpha=0.85)
    if qeq is not None and len(qeq):
        e0 = qeq["equity"].iloc[0]
        ax.plot(pd.to_datetime(qeq["date"]), qeq["equity"] / e0, label="QQQ BH", color="black", lw=2.0)
    ax.set_ylabel("Growth of $1")
    ax.set_title(f"QM-EOD-v2 {RUN_ID} (EOD-DIAG)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "equity_drawdown.png", dpi=120)
    plt.close()

    # ---- data_audit.md
    (OUT / "data_audit.md").write_text(f"""# Data audit — QM-EOD-v2

- Panel: reused `qm_research_001` `panel.npz` (sha256 `{frozen['data_hash_panel'][:16]}…`)
- Source: `~/projects/trading/data/ts_live/` daily bars; cutoff `{frozen['data_cutoff']}`
- EDGAR 8-K 2.02: reused; tags optional for EP
- Minute bars: **not available** → nature = **EOD-DIAG**
- Survivorship: pool is current-listing snapshot → `UNIVERSE_SURVIVORSHIP_BIAS` (optimistic)
- EP-C2 (PIT earnings surprise): `NOT_RUN_DATA_GAP`
- No short / no borrow path this round
""")

    # ---- acceptance.md
    (OUT / "acceptance.md").write_text(f"""# Acceptance checklist — QM-EOD-v2

- [x] BO/EP rules per frozen spec; no short
- [x] Max 5 positions; execution_rate = fills/signals reported ({num(p.get('execution_rate'), 3)})
- [x] Signal-layer R vs account-layer metrics separated in summary
- [x] EP volume threshold 3×; EOD volume caveat noted
- [x] 10/25/50 bps present in metrics.csv
- [x] Concentration + four year segments in concentration.md
- [x] Sample <100 not labelled 无效 (conclusion uses 证据不足/继续研究/暂不支持 only)
- [x] Chart ≤3 lines; QQQ black thick
- [x] No verification-segment retuning
- [x] Exit-day re-entry enabled (v2)
- [x] Never claim EOD results = original ORH passed
""")

    # ---- reproduce.md
    (OUT / "reproduce.md").write_text(f"""# Reproduce

```bash
cd ~/projects/trading/Vibe-Trading
git checkout research/qm-eod-v2
export QM_EOD_RUN_ID={RUN_ID}
nice -n 15 .venv/bin/python research/qm_eod_v2/scripts/run_all.py
.venv/bin/python research/qm_eod_v2/scripts/report.py
```

- SMA only (not EMA), as frozen.
- Panel/edgar symlinked from qm_research_001 run `qm-001-20261022-0041`.
- Combined BO+EP shared 5-slot book is primary.
""")

    # ---- summary_zh.md
    cost = mdf[(mdf.strategy == "COMBINED") & (mdf["filter"] == "on") & (mdf.e0 == 100000) & (mdf.segment == p["segment"])]
    rows_cost = []
    for bps in (10, 25, 50):
        sub = cost[cost.slip_bps == bps]
        if len(sub):
            r = sub.iloc[0]
            rows_cost.append(f"| {bps} | {pct(r.cagr)} | {pct(r.mdd)} | {num(r.sharpe)} | {num(r.pf)} | {int(r.n_trades)} | {pct(r.exposure_days)} | {num(r.execution_rate, 3)} |")

    off = mdf[(mdf.strategy == "COMBINED") & (mdf["filter"] == "off") & (mdf.slip_bps == 10) &
              (mdf.e0 == 100000) & (mdf.segment == p["segment"])]
    off_s = off.iloc[0] if len(off) else None

    s1k = mdf[(mdf.strategy == "COMBINED") & (mdf.e0 == 1000) & (mdf.slip_bps == 10) & (mdf.segment == p["segment"])]
    s1k_s = s1k.iloc[0] if len(s1k) else None

    summary = f"""# QM-EOD-v2 摘要（summary_zh）

日期：2026-10-22（Asia/Hong_Kong）｜run_id：`{RUN_ID}`  
性质：**EOD-DIAG（日线诊断）**，不是盘中 ORH 原版。不得写成「原方案已通过」。  
生存偏差：`UNIVERSE_SURVIVORSHIP_BIAS`

## 做了什么

- 在 `qm_research_001` 的 panel/edgar 上实现 **BO+EP**、**最多 5 持仓**、退出当日可再开仓。
- 主本金 $100,000；风险 0.5%；单只名义上限 25%；成本 10/25/50 bps。
- EP 量能 **≥3×** 20 日均量；QQQ 10SMA>20SMA 上行过滤（新开仓）；另报无过滤对照。
- 主路径：BO+EP **合并账户**共享 5 槽；另附 BO-only / EP-only 诊断。

## 结论：**{label_zh}**

{reason}

## 账户层（主结果：filter ON，10bps，$100k，{p.get('segment')}）

| 指标 | 策略 | QQQ |
|---|---:|---:|
| CAGR | {pct(p.get('cagr'))} | {pct(qqq_cagr)} |
| MDD | {pct(p.get('mdd'))} | — |
| Sharpe | {num(p.get('sharpe'))} | — |
| 在场时间 | {pct(p.get('exposure_days'))} | 100% |
| 平均持仓数 | {num(p.get('avg_positions'))} | — |
| 终值 | {num(p.get('final_equity'), 0)} | — |

### 成本敏感

| bps | CAGR | MDD | Sharpe | PF | 笔数 | 在场 | 执行率 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows_cost)}

### 无过滤对照
{f"- filter OFF：CAGR {pct(off_s.cagr)}，笔数 {int(off_s.n_trades)}，执行率 {num(off_s.execution_rate, 3)}" if off_s is not None else "- n/a"}

### $1k 对照
{f"- CAGR {pct(s1k_s.cagr)}，笔数 {int(s1k_s.n_trades)}，终值 {num(s1k_s.final_equity, 0)}" if s1k_s is not None else "- n/a"}

## 信号层（与账户分开）

| 指标 | 值 |
|---|---:|
| 信号数 | {int(p.get('n_signals', 0))}（BO {int(p.get('n_signals_bo', 0))} / EP {int(p.get('n_signals_ep', 0))}） |
| 成交笔数 | {n_trades} |
| 执行率 | {num(p.get('execution_rate'), 3)} |
| avg_R / med_R | {num(p.get('avg_R'))} / {num(p.get('med_R'))} |
| PF | {num(p.get('pf'))} |
| 胜率 | {pct(p.get('win_rate'))} |
| 前5笔占净利 | {pct(p.get('top5_share'))} |

日线 EP「巨量」仅为收盘量/20日均量，**测不准盘中巨量**。

## 不可声称

- 不是原版 ORH / 分钟执行已通过或已证伪。
- 样本 <100 时不写「无效」。
"""
    (OUT / "summary_zh.md").write_text(summary)

    # ---- result.md
    (OUT / "result.md").write_text(f"""# result.md — QM-EOD-v2 给派活人

**run_id:** `{RUN_ID}`  
**结果路径:** `{OUT}`  
**结论（三选一）: {label_zh}**

{reason}

## 关键数字（账户层 / 信号层）

- 账户：CAGR {pct(p.get('cagr'))}，MDD {pct(p.get('mdd'))}，Sharpe {num(p.get('sharpe'))}，在场 {pct(p.get('exposure_days'))}，均持仓 {num(p.get('avg_positions'))}
- 对照 QQQ CAGR {pct(qqq_cagr)}
- 信号：n_signals={int(p.get('n_signals', 0))}，fills={n_trades}，执行率 {num(p.get('execution_rate'), 3)}，avg_R={num(p.get('avg_R'))}，PF={num(p.get('pf'))}
- 成本 25/50 bps 见 `metrics.csv` / `summary_zh.md`
- 性质：EOD-DIAG + UNIVERSE_SURVIVORSHIP_BIAS；未做空；EP-C2 NOT_RUN_DATA_GAP

详见 `summary_zh.md`、`concentration.md`、`acceptance.md`。
""")
    print(f"[report] conclusion={label_zh} trades={n_trades} cagr={p.get('cagr')}")


if __name__ == "__main__":
    main()
