"""v1.1.1 vs cash_floor=0.20 / gross_cap=0.80. Research only.

Uses the official run_portfolio. The position-scan copy is only the zero-diff
check at canonical parameters. Does not edit upleg_bt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HANDOFF = Path("/tmp/ca-v111/handoff/repo_handoff_backtest_20261010T185418+0800")
DATA = Path("/tmp/ca-v111/upleg_work/data")
sys.path.insert(0, str(HANDOFF))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upleg-v111-position-scan-20261025"))

from upleg_bt.engine.fm import FM
from upleg_bt.engine.portfolio import run_portfolio
from upleg_bt.metrics import headline, perf

from scan_run import run_scan

OUT = Path(__file__).resolve().parent
END = "2026-09-30"
START_2019 = "2019-01-02"
SNAP_SHA = "39a415dc2f8ebfadc30634cc1e042483ed3f6c0b457add20c8e92a47abc96f21"
HANDOFF_SHA = "f76825cde54270ff0258793fd89fdd490a285924a1a6cf65f69b4da67c9f0602"

PERIODS = {
    "2019_plus": (START_2019, END),
    "2019_2022": (START_2019, "2022-12-31"),
    "2023_plus": ("2023-01-01", END),
}


def window_metrics(df: pd.DataFrame, a: str, b: str) -> dict:
    m = perf(df, a, b)
    sl = df.loc[a:b]
    m["trade_days"] = int(sl["fee"].gt(0).sum())
    eq = sl["equity"]
    dd = eq / eq.cummax() - 1.0
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    m["maxdd_peak"] = str(pd.Timestamp(peak).date())
    m["maxdd_trough"] = str(pd.Timestamp(trough).date())
    m["equity_start"] = float(eq.iloc[0])
    m["equity_end"] = float(eq.iloc[-1])
    return m


def flat_row(name: str, params: dict, book: dict, long_df: pd.DataFrame, long_a: str) -> dict:
    row = {"name": name, **params, "long_start": long_a}
    for label, (a, b) in PERIODS.items():
        m = window_metrics(book, a, b)
        for k, v in m.items():
            row[f"{label}_{k}"] = v
    mlong = window_metrics(long_df, long_a, END)
    for k, v in mlong.items():
        row[f"long_{k}"] = v
    return row


def drawdown_point(eq: pd.Series) -> dict:
    dd = eq / eq.cummax() - 1.0
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    return {
        "maxdd": float(dd.min()),
        "peak": str(pd.Timestamp(peak).date()),
        "trough": str(pd.Timestamp(trough).date()),
        "equity_at_trough": float(eq.loc[trough]),
        "equity_at_peak": float(eq.loc[peak]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("LOAD", flush=True)
    D = FM(str(DATA), processes=2)
    long_a = str(pd.Timestamp(D.cal[1]).date())
    cal0 = str(pd.Timestamp(D.cal[0]).date())
    cal_last = str(pd.Timestamp(D.cal[-1]).date())
    print(json.dumps({"cal0": cal0, "long_a": long_a, "cal_last": cal_last, "T": int(D.T), "N": int(D.N)}), flush=True)

    print("OFFICIAL_2019", flush=True)
    official = run_portfolio(D, record=False)
    print("SCAN_COPY", flush=True)
    mine = run_scan(D, START_2019, END)
    same = len(official["eq"]) == len(mine) and bool((official["eq"].index == mine.index).all())
    maxabs = float(np.max(np.abs(official["eq"]["equity"].to_numpy() - mine["equity"].to_numpy()))) if same else None
    exp_abs = float(np.max(np.abs(official["eq"]["exposure"].to_numpy() - mine["exposure"].to_numpy()))) if same else None
    fee_abs = float(np.max(np.abs(official["eq"]["fee"].to_numpy() - mine["fee"].to_numpy()))) if same else None
    m = perf(official["eq"])
    trade_days = int(official["eq"]["fee"].gt(0).sum())
    h = headline(m)
    verify = {
        "snapshot_sha256": SNAP_SHA,
        "handoff_sha256": HANDOFF_SHA,
        "calendar_first": cal0,
        "calendar_last": cal_last,
        "long_start": long_a,
        "engine": "upleg_bt.engine.portfolio.run_portfolio",
        "scanner_copy": "results/upleg-v111-position-scan-20261025/scan_run.py",
        "same_index": same,
        "equity_maxabs": maxabs,
        "exposure_maxabs": exp_abs,
        "fee_maxabs": fee_abs,
        "official": {**m, "trade_days": trade_days, "headline": h},
        "expected_headline": {"cagr": "18.0%", "maxdd": "-11.7%", "sortino": "1.93", "sharpe": "1.33"},
        "PASS": bool(
            same
            and maxabs is not None
            and maxabs < 1e-9
            and exp_abs < 1e-9
            and fee_abs < 1e-9
            and h == {"cagr": "18.0%", "maxdd": "-11.7%", "sortino": "1.93", "sharpe": "1.33"}
            and abs(m["avg_exposure"] - 0.5914710808789265) < 1e-6
            and trade_days == 496
        ),
    }
    (OUT / "verify.json").write_text(json.dumps(verify, indent=1, ensure_ascii=False))
    print(json.dumps({"PASS": verify["PASS"], "headline": h, "trade_days": trade_days, "equity_maxabs": maxabs, "avg_exposure": m["avg_exposure"]}), flush=True)
    if not verify["PASS"]:
        raise SystemExit("baseline does not match locked v1.1.1")

    base_kw = dict(K=3, reserve=0.30, gross_cap=0.70, vt=0.16, record=False)
    cand_kw = dict(K=3, reserve=0.20, gross_cap=0.80, vt=0.16, record=False)
    print("CANDIDATE_2019", flush=True)
    cand = run_portfolio(D, **cand_kw)
    print("LONG", flush=True)
    base_long = run_portfolio(D, start=long_a, end=END, **base_kw)
    cand_long = run_portfolio(D, start=long_a, end=END, **cand_kw)

    # The 2019 official call above used defaults (gross_cap None → 0.70). Re-run with
    # the explicit pair so the baseline row is the same call shape as the candidate.
    base = run_portfolio(D, **base_kw)
    explicit_maxabs = float(np.max(np.abs(base["eq"]["equity"].to_numpy() - official["eq"]["equity"].to_numpy())))
    if explicit_maxabs >= 1e-9:
        raise SystemExit(f"explicit gross_cap=0.70 drifted from default: {explicit_maxabs}")

    rows = [
        flat_row("v1.1.1", {"cash_floor": 0.30, "gross_cap": 0.70, "vol_target": 0.16}, base["eq"], base_long["eq"], long_a),
        flat_row("upleg_v111_cash20_gross80", {"cash_floor": 0.20, "gross_cap": 0.80, "vol_target": 0.16}, cand["eq"], cand_long["eq"], long_a),
    ]
    keep = [
        "name", "cash_floor", "gross_cap", "vol_target", "long_start",
    ]
    metric_keys = ["cagr", "maxdd", "sharpe", "sortino", "avg_exposure", "ann_vol", "trade_days"]
    for label in ("2019_plus", "2019_2022", "2023_plus", "long"):
        for k in metric_keys:
            keep.append(f"{label}_{k}")
    table = pd.DataFrame(rows)[keep]
    table.to_csv(OUT / "comparison.csv", index=False, float_format="%.10g")

    detail = {
        "baseline_2019_drawdown": drawdown_point(base["eq"]["equity"]),
        "candidate_2019_drawdown": drawdown_point(cand["eq"]["equity"]),
        "baseline_long_drawdown": drawdown_point(base_long["eq"]["equity"]),
        "candidate_long_drawdown": drawdown_point(cand_long["eq"]["equity"]),
        "explicit_gross_cap_maxabs": explicit_maxabs,
        "rows": rows,
    }
    (OUT / "detail.json").write_text(json.dumps(detail, indent=1, ensure_ascii=False, default=str))

    hb = headline(perf(base["eq"]))
    hc = headline(perf(cand["eq"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
    plt.rcParams["font.family"] = "WenQuanYi Micro Hei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11.2, 6.4), dpi=150)
    b_eq = base["eq"]["equity"] / base["eq"]["equity"].iloc[0]
    c_eq = cand["eq"]["equity"] / cand["eq"]["equity"].iloc[0]
    ax.plot(
        b_eq.index, b_eq, color="black", linewidth=2.6, linestyle="-",
        label=f"v1.1.1 现金30%总仓70%（年化 {hb['cagr']}，最大回撤 {hb['maxdd']}，索提诺 {hb['sortino']}）",
    )
    ax.plot(
        c_eq.index, c_eq, color="#D55E00", linewidth=2.0, linestyle=(0, (7, 4)),
        label=f"候选 现金20%总仓80%（年化 {hc['cagr']}，最大回撤 {hc['maxdd']}，索提诺 {hc['sortino']}）",
    )
    ax.set_title("2019-01-02 至 2026-09-30 净值：v1.1.1 与现金 20% / 总仓 80%")
    ax.set_ylabel("净值（起点 1）")
    ax.set_xlabel("日期")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "equity_v111_vs_cash20_gross80.png")
    plt.close()
    verify["chart"] = "equity_v111_vs_cash20_gross80.png"
    verify["candidate"] = {"cash_floor": 0.20, "gross_cap": 0.80, "vol_target": 0.16}
    verify["explicit_gross_cap_maxabs"] = explicit_maxabs
    verify["long_drawdown"] = {
        "baseline": detail["baseline_long_drawdown"],
        "candidate": detail["candidate_long_drawdown"],
    }
    (OUT / "verify.json").write_text(json.dumps(verify, indent=1, ensure_ascii=False))
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
