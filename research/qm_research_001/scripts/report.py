"""Build acceptance.csv, statistics.md, attribution.md and 00_summary_zh.md from run outputs."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qm import metrics  # noqa: E402

RUN = "runs/qm-001-20261022-0041"
RES = f"{RUN}/deliverables"
STRATS = {"QM-BO-001": "BO", "QM-EP-001": "EP", "QM-PS-001": "PS"}
MAIN = {"BO": "BO-MAIN", "EP": "EP-MAIN", "PS": "PS-MAIN"}
SEGS = ["CONT", "S1", "S2", "S3"]


def load():
    comp = pd.read_csv(f"{RES}/comparisons.csv")
    bench = pd.read_csv(f"{RES}/benchmarks/benchmark_metrics.csv")
    stress = pd.read_csv(f"{RES}/stress.csv") if os.path.exists(f"{RES}/stress.csv") else pd.DataFrame()
    borrow = pd.read_csv(f"{RES}/borrow_scenarios.csv") if os.path.exists(f"{RES}/borrow_scenarios.csv") else pd.DataFrame()
    stats = pd.read_csv(f"{RES}/statistics.csv") if os.path.exists(f"{RES}/statistics.csv") else pd.DataFrame()
    return comp, bench, stress, borrow, stats


def row(comp, strat, cfg, seg):
    d = comp[(comp.strategy == strat) & (comp.config == cfg) & (comp.segment == seg)]
    return d.iloc[0] if len(d) else None


def brow(bench, name, seg):
    d = bench[(bench.benchmark == name) & (bench.segment == seg)]
    return d.iloc[0] if len(d) else None


def accept(comp, bench, stress, stats):
    rows = []
    verdicts = {}
    for sid, s in STRATS.items():
        main = MAIN[s]
        c = {seg: row(comp, s, main, seg) for seg in SEGS}
        q = {seg: brow(bench, "QQQ-BH100", seg) for seg in SEGS}
        q30 = {seg: brow(bench, "QQQ-30", seg) for seg in SEGS}
        sp30 = {seg: brow(bench, "SPY-30", seg) for seg in SEGS}

        def add(layer, cond, value, threshold, ok, note=""):
            rows.append({"strategy": sid, "layer": layer, "condition": cond, "value": value,
                         "threshold": threshold, "pass": ok, "note": note})

        cont = c["CONT"]
        add("A", "A1 连续验证净期望R>0", cont.avg_R, ">0", bool(cont.avg_R > 0))
        add("A", "A2 连续验证PF>=1.20", cont.pf, ">=1.20", bool(cont.pf >= 1.20))
        add("A", "A3 连续验证CAGR>0", cont.cagr, ">0", bool(cont.cagr > 0))
        add("A", "A4 连续验证Sharpe>0", cont.sharpe, ">0", bool(cont.sharpe > 0))
        add("A", "A5 连续验证日回撤<=30%", abs(cont.mdd), "<=0.30", bool(abs(cont.mdd) <= 0.30))
        add("A", "A6 正常完成交易>=100", cont.n_trades, ">=100", bool(cont.n_trades >= 100))
        add("A", "A7 不同证券>=30", cont.unique_syms, ">=30", bool(cont.unique_syms >= 30))
        segs_n = [c[x].n_trades for x in ["S1", "S2", "S3"]]
        add("A", "A8 三独立分段各>=20笔", "/".join(str(int(x)) for x in segs_n), "each>=20",
            bool(min(segs_n) >= 20))
        seg_pos = [int(c[x].avg_R > 0) for x in ["S1", "S2", "S3"]]
        add("A", "A9 至少2/3分段净期望R>0", f"{sum(seg_pos)}/3", ">=2/3", bool(sum(seg_pos) >= 2))
        add("A", "A10 无分段回撤>30%", f"{max(abs(c[x].mdd) for x in ['S1','S2','S3']):.3f}",
            "<=0.30", bool(max(abs(c[x].mdd) for x in ["S1", "S2", "S3"]) <= 0.30))

        st = stress[stress.config == f"{main}-SLIP25"] if len(stress) else pd.DataFrame()
        if len(st):
            v = st[st.segment == "CONT"].iloc[0]
            add("A", "A11 25bps净期望R>0且PF>1", f"R={v.avg_R:.3f},PF={v.pf:.2f}", "R>0,PF>1",
                bool(v.avg_R > 0 and v.pf > 1))
        else:
            add("A", "A11 25bps场景", "NOT_RUN", "R>0,PF>1", None)
        add("A", "A12 延迟1分钟场景", "NOT_APPLICABLE（EOD 诊断无分钟执行层）", "R>0,PF>1", None)
        pert = comp[(comp.strategy == s) & (comp.segment == "CONT") & comp.config.str.startswith(f"{s}-P")]
        npos = int((pert.avg_R > 0).sum())
        add("A", "A13 8个扰动中>=6个净期望R>0", f"{npos}/{len(pert)}", ">=6/8", bool(npos >= 6))
        bs = stats[(stats.strategy == s) & (stats.metric == "trade_R_mean")]
        if len(bs):
            add("A", "A14 交易R均值区块bootstrap下界>0", bs.iloc[0]["lo"], ">0", bool(bs.iloc[0]["lo"] > 0))
        else:
            add("A", "A14 交易R bootstrap", "NO_DATA", ">0", None)

        # B layer vs QQQ-BH100
        d_cagr = cont.cagr - q["CONT"].cagr
        d_mdd = abs(cont.mdd) - abs(q["CONT"].mdd)
        d_sharpe = cont.sharpe - q["CONT"].sharpe
        add("B", "B1 连续验证CAGR高于QQQ>=1pp", round(d_cagr, 4), ">=0.01", bool(d_cagr >= 0.01))
        add("B", "B2 回撤幅度小于QQQ>=2pp", round(-d_mdd, 4), ">=0.02", bool(d_mdd <= -0.02))
        add("B", "B3 Sharpe高于QQQ>=0.10", round(d_sharpe, 4), ">=0.10", bool(d_sharpe >= 0.10))
        add("B", "B4 自身CAGR>0且Sharpe>0且回撤<=30%",
            f"{cont.cagr:.3f}/{cont.sharpe:.2f}/{cont.mdd:.3f}", "all", bool(cont.cagr > 0 and cont.sharpe > 0 and abs(cont.mdd) <= 0.30))
        wins = 0
        for seg in ["S1", "S2", "S3"]:
            a, b = c[seg], q[seg]
            if (a.cagr - b.cagr >= 0.01) and (abs(a.mdd) <= abs(b.mdd) - 0.02) and (a.sharpe - b.sharpe >= 0.10):
                wins += 1
        add("B", "B5 三独立分段>=2段三指标严格改善", f"{wins}/3", ">=2/3", bool(wins >= 2))
        a_true = all(r["pass"] is True for r in rows if r["strategy"] == sid and r["layer"] == "A")
        add("B", "B6 A层同时成立", a_true, "true", bool(a_true))
        dom = (q30["CONT"].cagr > cont.cagr) and (abs(q30["CONT"].mdd) < abs(cont.mdd)) and (q30["CONT"].sharpe > cont.sharpe)
        dom2 = (sp30["CONT"].cagr > cont.cagr) and (abs(sp30["CONT"].mdd) < abs(cont.mdd)) and (sp30["CONT"].sharpe > cont.sharpe)
        add("B", "B7 未被QQQ-30/SPY-30三指标同时压制", f"QQQ30={not dom},SPY30={not dom2}", "both true",
            bool((not dom) and (not dom2)))

        # layer verdicts
        a_rows = [r for r in rows if r["strategy"] == sid and r["layer"] == "A"]
        a_pass = [r["pass"] for r in a_rows]
        if any(p is None for p in a_pass):
            av = "INCONCLUSIVE"
        elif all(a_pass):
            av = "SIGNAL_SUPPORTED"
        else:
            av = "SIGNAL_NOT_SUPPORTED"
        verdicts[sid] = {"A": av, "B": "INCOMPLETE", "C": "EXECUTION_INCOMPLETE",
                         "n_trades_CONT": int(cont.n_trades), "cagr": float(cont.cagr),
                         "mdd": float(cont.mdd), "sharpe": float(cont.sharpe),
                         "avg_R": float(cont.avg_R), "pf": float(cont.pf)}
    return pd.DataFrame(rows), verdicts


def main():
    comp, bench, stress, borrow, stats = load()
    acc, verdicts = accept(comp, bench, stress, stats)
    acc.to_csv(f"{RES}/acceptance.csv", index=False)
    json.dump(verdicts, open(f"{RUN}/deliverables/verdicts.json", "w"), indent=1, default=float)

    # attribution: paired daily net returns main vs controls
    attr = []
    pairs = {"BO": ["BO-C1", "BO-C2", "BO-C3"], "EP": ["EP-C1", "EP-C3"], "PS": ["PS-C1", "PS-C2", "PS-C3"]}
    for s, ctrls in pairs.items():
        base = pd.read_csv(f"{RES}/{s}/{MAIN[s]}/CONT_equity.csv")
        rb = base["equity"].pct_change().to_numpy()[1:]
        for ctl in ctrls:
            p = f"{RES}/{s}/{ctl}/CONT_equity.csv"
            if not os.path.exists(p):
                continue
            e = pd.read_csv(p)
            rc = e["equity"].pct_change().to_numpy()[1:]
            n = min(len(rb), len(rc))
            bs = metrics.block_bootstrap(rb[:n], rc[:n], block=20, reps=2000)
            attr.append({"strategy": s, "comparison": f"{MAIN[s]} vs {ctl}", **bs})
    pd.DataFrame(attr).to_csv(f"{RES}/attribution.csv", index=False)

    # comparison table
    tbl = []
    for sid, s in STRATS.items():
        r = verdicts[sid]
        tbl.append({"策略": sid, "主版本": r["A"], "B": r["B"], "C": r["C"],
                    "交易数": r["n_trades_CONT"], "CAGR": f"{r['cagr']*100:.2f}%",
                    "最大回撤": f"{r['mdd']*100:.2f}%", "Sharpe": f"{r['sharpe']:.2f}",
                    "净期望R": f"{r['avg_R']:.3f}", "PF": f"{r['pf']:.2f}"})
    for nm in ["QQQ-BH100", "SPY-BH100", "QQQ-30", "SPY-30"]:
        b = brow(bench, nm, "CONT")
        tbl.append({"策略": nm, "主版本": "—", "B": "—", "C": "—", "交易数": "—",
                    "CAGR": f"{b.cagr*100:.2f}%", "最大回撤": f"{b.mdd*100:.2f}%",
                    "Sharpe": f"{b.sharpe:.2f}", "净期望R": "—", "PF": "—"})
    pd.DataFrame(tbl).to_csv(f"{RES}/comparison_table.csv", index=False)
    print(pd.DataFrame(tbl).to_string(index=False))
    print()
    print(acc[acc.layer == "A"][["strategy", "condition", "value", "pass"]].to_string(index=False))
    print()
    print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()
