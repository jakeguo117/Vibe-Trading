"""Run the locked round-2 gates and write tables, charts, and the Chinese note."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COMMODITY, CRYPTO, LEV_MAP, STOCKS, load_stooq, load_yahoo, make_pack, probe_nasdaq
from .engine import cfg_control, cfg_momentum, run_backtest, slice_metrics
from .plots import line_chart

ROOT = Path(__file__).resolve().parents[1]
ART = Path("/opt/cursor/artifacts")


def load_thresholds() -> dict:
    path = ROOT / "thresholds.json"
    raw = path.read_bytes()
    doc = json.loads(raw)
    doc["_sha256"] = hashlib.sha256(raw).hexdigest()
    doc["_path"] = str(path)
    return doc


def _row(group, candidate, spec, result, extra=None):
    r = {
        "group": group,
        "candidate": candidate,
        "spec": spec,
        "start": result.get("start", ""),
        "end": result.get("end", ""),
        "cagr": result.get("cagr"),
        "total_return": result.get("total"),
        "max_dd": result.get("maxdd"),
        "worst_year": result.get("worst_year"),
        "worst_year_label": result.get("worst_year_label", ""),
        "sharpe": result.get("sharpe"),
        "sortino": result.get("sortino"),
        "ann_vol": result.get("vol"),
        "n_days": result.get("n_days"),
        "n_trades": result.get("n_trades"),
        "max_exposure": result.get("max_exposure"),
        "oos_cagr": result.get("oos_cagr"),
        "oos_max_dd": result.get("oos_max_dd"),
        "pass_beats_oos": result.get("pass_beats_oos"),
        "pass_maxdd": result.get("pass_maxdd"),
        "pass_worst_year": result.get("pass_worst_year"),
        "pass_perturb": result.get("pass_perturb"),
        "pass_walkforward": result.get("pass_walkforward"),
        "pass_all": result.get("pass_all"),
        "lev_days": result.get("lev_days"),
        "detail": result.get("detail", ""),
    }
    if extra:
        r.update(extra)
    return r


def _attach_oos(result, oos_start, min_days):
    oos = slice_metrics(result["curve"].dropna(), oos_start, result["curve"].dropna().index.max(), min_days)
    result["oos_cagr"] = oos["cagr"]
    result["oos_max_dd"] = oos["maxdd"]
    result["oos_worst_year"] = oos["worst_year"]
    return result


def _primary_flags(result, control_oos, thresholds):
    rules = thresholds["pass"]
    beats = bool(np.isfinite(result.get("oos_cagr", np.nan)) and np.isfinite(control_oos) and result["oos_cagr"] > control_oos)
    dd_ok = bool(np.isfinite(result["maxdd"]) and result["maxdd"] >= rules["max_drawdown_floor"])
    yr_ok = bool(np.isfinite(result["worst_year"]) and result["worst_year"] >= rules["worst_calendar_year_floor"])
    result["pass_beats_oos"] = beats
    result["pass_maxdd"] = dd_ok
    result["pass_worst_year"] = yr_ok
    return beats, dd_ok, yr_ok


def _drawdown(curve: pd.Series) -> pd.Series:
    c = curve.dropna()
    peak = c.cummax()
    return c / peak - 1.0


def _stitch(curves: list[pd.Series]) -> pd.Series:
    pieces = []
    level = 1.0
    for curve in curves:
        c = curve.dropna()
        if len(c) < 2:
            continue
        rebased = c / float(c.iloc[0]) * level
        pieces.append(rebased.iloc[1:] if pieces else rebased)
        level = float(rebased.iloc[-1])
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def _monte_carlo(trades, seed: int, n_sims: int = 2000) -> dict:
    if len(trades) < 8:
        return {"n_trades": len(trades), "note": "成交太少，不估计分布"}
    rets = np.array([t[1] for t in trades], dtype=np.float64)
    years = [str(t[0])[:4] for t in trades]
    order = sorted(set(years))
    counts = [years.count(y) for y in order]
    rng = np.random.default_rng(seed)
    dds = np.empty(n_sims)
    any_bad = 0
    bad_years = 0
    year_n = 0
    for i in range(n_sims):
        sample = rng.choice(rets, size=len(rets), replace=True)
        eq = np.cumprod(1.0 + sample)
        peak = np.maximum.accumulate(eq)
        dds[i] = float((eq / peak - 1.0).min())
        cursor = 0
        hit = False
        for k in counts:
            chunk = sample[cursor : cursor + k]
            cursor += k
            year_ret = float(np.prod(1.0 + chunk) - 1.0) if len(chunk) else 0.0
            year_n += 1
            if year_ret < -0.30:
                bad_years += 1
                hit = True
        any_bad += int(hit)
    return {
        "n_trades": int(len(rets)),
        "n_sims": n_sims,
        "seed": seed,
        "maxdd_p05": float(np.quantile(dds, 0.05)),
        "maxdd_p50": float(np.quantile(dds, 0.50)),
        "maxdd_p95": float(np.quantile(dds, 0.95)),
        "p_sim_has_year_loss_gt_30": any_bad / n_sims,
        "p_year_loss_gt_30": bad_years / year_n if year_n else float("nan"),
    }


def _block_bootstrap_dd(curve: pd.Series, seed: int, n_sims: int = 2000, block: int = 21) -> dict:
    r = curve.dropna().pct_change().dropna().to_numpy()
    if len(r) < block * 4:
        return {}
    rng = np.random.default_rng(seed + 1)
    n = len(r)
    n_blocks = int(math.ceil(n / block))
    dds = np.empty(n_sims)
    for i in range(n_sims):
        starts = rng.integers(0, n - block, size=n_blocks)
        path = np.concatenate([r[s : s + block] for s in starts])[:n]
        eq = np.cumprod(1.0 + path)
        peak = np.maximum.accumulate(eq)
        dds[i] = float((eq / peak - 1.0).min())
    return {"block": block, "maxdd_p50": float(np.quantile(dds, 0.50)), "maxdd_p05": float(np.quantile(dds, 0.05))}


def _select_fold(pack, base, grid, train_start, train_end, min_days, rules):
    best = None
    for mom, vt, ma in grid:
        cfg = replace(base, mom_n=mom, vt=vt, ma=ma)
        res = run_backtest(pack, cfg, train_start, train_end, min_days)
        feasible = (
            np.isfinite(res["maxdd"])
            and res["maxdd"] >= rules["max_drawdown_floor"]
            and (not np.isfinite(res["worst_year"]) or res["worst_year"] >= rules["worst_calendar_year_floor"])
        )
        sharpe = res["sharpe"] if np.isfinite(res["sharpe"]) else -1e9
        cagr = res["cagr"] if np.isfinite(res["cagr"]) else -1e9
        key = (1 if feasible else 0, sharpe, cagr)
        if best is None or key > best[0]:
            best = (key, cfg, bool(feasible), res)
    return best


def run_study():
    thresholds = load_thresholds()
    opt = thresholds["option_approx"]
    min_days = int(thresholds["pass"]["min_days_for_calendar_year"])
    seed = int(thresholds["seed"])
    cache = ROOT / "cache"
    print("loading yahoo panel...", flush=True)
    panel = load_yahoo(cache)
    pack = make_pack(panel)
    print(f"panel names={pack.n} {pack.dates[0].date()} → {pack.dates[-1].date()}", flush=True)
    primary_start = thresholds["primary_start"]
    long_start = thresholds["long_start"]
    oos_start = thresholds["oos_start"]
    end = str(pack.dates[-1].date())

    def run(cfg, start=None, stop=None):
        if start is None:
            start = primary_start
        if stop is None:
            stop = end
        cfg = replace(cfg, opt=opt, cell_min=int(thresholds["hist_score"]["cell_min_segments"]))
        return run_backtest(pack, cfg, start, stop, min_days)

    print("primary control / S / SC", flush=True)
    control = run(cfg_control(), primary_start)
    _attach_oos(control, oos_start, min_days)
    control_oos = control["oos_cagr"]
    base_S = cfg_momentum("S", "S")
    base_SC = cfg_momentum("SC", "SC")
    runs = {
        "control": control,
        "S": _attach_oos(run(base_S), oos_start, min_days),
        "SC": _attach_oos(run(base_SC), oos_start, min_days),
    }
    for key in ("S", "SC"):
        _primary_flags(runs[key], control_oos, thresholds)

    cash_runs = {}
    for cash in (0.10, 0.30):
        for uni, base in (("S", base_S), ("SC", base_SC)):
            name = f"{uni}_cash{int(cash*100)}"
            cash_runs[name] = _attach_oos(run(replace(base, name=name, cash_floor=cash)), oos_start, min_days)
            _primary_flags(cash_runs[name], control_oos, thresholds)

    cap_runs = {}
    for cap in (0.0, 0.20, 0.40):
        name = f"SC_cryptocap{int(cap*100)}"
        cap_runs[name] = _attach_oos(
            run(replace(base_SC, name=name, crypto_cap=cap)),
            oos_start,
            min_days,
        )
        _primary_flags(cap_runs[name], control_oos, thresholds)

    print("long history", flush=True)
    long_runs = {
        "control": run(cfg_control(), long_start),
        "S": run(base_S, long_start),
        "SC": run(base_SC, long_start),
        "control_lev": run(replace(cfg_control(), name="control_lev", deriv="lev"), long_start),
    }

    print("derivatives", flush=True)
    deriv_runs = {}
    for uni, base in (("control", cfg_control()), ("S", base_S), ("SC", base_SC)):
        for deriv in ("lev", "leap", "spread"):
            name = f"{uni}_{deriv}"
            deriv_runs[name] = _attach_oos(run(replace(base, name=name, deriv=deriv)), oos_start, min_days)
            _primary_flags(deriv_runs[name], control_oos, thresholds)

    print("perturbation", flush=True)
    shocks = {
        "mom_window": (("mom_n", 16), ("mom_n", 24)),
        "vol_target": (("vt", 0.20), ("vt", 0.30)),
        "ma_length": (("ma", 24), ("ma", 36)),
    }
    perturb = {}
    for label, base, res0 in (("S", base_S, runs["S"]), ("SC", base_SC, runs["SC"])):
        base_ann = res0["cagr"]
        ok = bool(np.isfinite(base_ann) and base_ann > 0)
        details = []
        for param, pairs in shocks.items():
            for field, value in pairs:
                cfg = replace(base, **{field: value}, name=f"{label}_{param}_{value}")
                res = run(cfg)
                ratio = res["cagr"] / base_ann if ok and base_ann else float("nan")
                survived = bool(ok and np.isfinite(ratio) and ratio >= thresholds["pass"]["perturbation_ann_ratio_floor"])
                if not survived:
                    ok = False
                details.append((cfg.name, res["cagr"], ratio, survived))
                perturb[cfg.name] = res
        res0["pass_perturb"] = ok
        res0["perturb_detail"] = details

    # Control derivatives use the same ±20% idea on the control's own knobs.
    # Momentum length does not enter the control rank, so those two shocks match the base.
    print("control-derivative perturbation", flush=True)
    for deriv in ("lev", "leap", "spread"):
        name = f"control_{deriv}"
        base_cfg = replace(cfg_control(), name=name, deriv=deriv)
        base_ann = deriv_runs[name]["cagr"]
        ok = bool(np.isfinite(base_ann) and base_ann > 0)
        details = []
        control_shocks = {
            "mom_window": (("mom_n", 16), ("mom_n", 24)),
            "vol_target": (("vt", 0.128), ("vt", 0.192)),
            "ma_length": (("ma", 24), ("ma", 36)),
        }
        for param, pairs in control_shocks.items():
            for field, value in pairs:
                cfg = replace(base_cfg, **{field: value}, name=f"{name}_{param}_{value}")
                res = run(cfg)
                ratio = res["cagr"] / base_ann if ok and base_ann else float("nan")
                survived = bool(ok and np.isfinite(ratio) and ratio >= thresholds["pass"]["perturbation_ann_ratio_floor"])
                if not survived:
                    ok = False
                details.append((cfg.name, res["cagr"], ratio, survived))
                perturb[cfg.name] = res
        deriv_runs[name]["pass_perturb"] = ok
        deriv_runs[name]["perturb_detail"] = details

    print("walk-forward", flush=True)
    grid = [
        (m, v, a)
        for m in thresholds["walk_forward"]["grid"]["mom_window"]
        for v in thresholds["walk_forward"]["grid"]["vol_target"]
        for a in thresholds["walk_forward"]["grid"]["ma_length"]
    ]
    wf_rules = thresholds["walk_forward"]["train_constraints"]
    wf = {}
    last_year = int(pd.Timestamp(pack.dates[-1]).year)
    for label, base in (("S", base_S), ("SC", base_SC)):
        fold_curves = []
        fold_rows = []
        chosen_by_year = {}
        for year in range(2001, last_year + 1):
            train_start, train_end = f"{year-3}-01-01", f"{year-1}-12-31"
            test_start, test_end = f"{year}-01-01", f"{year}-12-31"
            if pd.Timestamp(test_start) > pack.dates[-1]:
                continue
            best = _select_fold(pack, base, grid, train_start, train_end, min_days, wf_rules)
            cfg = best[1]
            chosen_by_year[year] = {
                "mom_n": cfg.mom_n,
                "vt": cfg.vt,
                "ma": cfg.ma,
                "train_feasible": best[2],
                "train_sharpe": best[3]["sharpe"],
            }
            test = run(cfg, test_start, min(test_end, end))
            fold_curves.append(test["curve"])
            fold_rows.append({"year": year, "cagr": test["cagr"], "maxdd": test["maxdd"], **chosen_by_year[year]})
            print(f"  WF {label} {year} mom={cfg.mom_n} vt={cfg.vt} ma={cfg.ma} test={test['cagr']}", flush=True)
        stitched = _stitch(fold_curves)
        sm = slice_metrics(stitched, stitched.index.min() if len(stitched) else "2001-01-01", end, min_days) if len(stitched) else {}
        positive = bool(sm and np.isfinite(sm.get("cagr", np.nan)) and sm["cagr"] > thresholds["pass"]["walk_forward_stitched_cagr_floor"])
        runs[label]["pass_walkforward"] = positive
        wf[label] = {"stitched": stitched, "metrics": sm, "folds": fold_rows, "positive": positive}

    for key in ("S", "SC"):
        flags = [
            runs[key].get("pass_beats_oos"),
            runs[key].get("pass_maxdd"),
            runs[key].get("pass_worst_year"),
            runs[key].get("pass_perturb"),
            runs[key].get("pass_walkforward"),
        ]
        runs[key]["pass_all"] = all(bool(x) for x in flags)

    # Derivative and cash-floor rows inherit walk-forward only when their own
    # primary gates are being judged: re-apply the selected params with the overlay.
    print("walk-forward overlays", flush=True)
    for store in (cash_runs, cap_runs, deriv_runs):
        for name, res in store.items():
            if name.startswith("control_"):
                curves = []
                deriv = name.split("_", 1)[1]
                for fold in wf["S"]["folds"]:
                    year = fold["year"]
                    cfg = replace(cfg_control(), name=name, deriv=deriv)
                    test = run(cfg, f"{year}-01-01", min(f"{year}-12-31", end))
                    curves.append(test["curve"])
                stitched = _stitch(curves)
                sm = slice_metrics(stitched, stitched.index.min(), end, min_days) if len(stitched) else {}
                res["pass_walkforward"] = bool(sm and np.isfinite(sm.get("cagr", np.nan)) and sm["cagr"] > 0)
                res["pass_all"] = all(
                    bool(res.get(k))
                    for k in ("pass_beats_oos", "pass_maxdd", "pass_worst_year", "pass_perturb", "pass_walkforward")
                )
                res["wf_cagr"] = sm.get("cagr") if sm else float("nan")
                continue
            parent = "SC" if name.startswith("SC") else "S"
            curves = []
            for fold in wf[parent]["folds"]:
                base = base_S if parent == "S" else base_SC
                cfg = replace(base, mom_n=fold["mom_n"], vt=fold["vt"], ma=fold["ma"])
                if name.endswith("lev") or name.endswith("leap") or name.endswith("spread"):
                    cfg = replace(cfg, deriv=name.split("_", 1)[1])
                if "cash" in name:
                    cash = int(name.split("cash")[1]) / 100.0
                    cfg = replace(cfg, cash_floor=cash)
                if "cryptocap" in name:
                    cap = int(name.split("cryptocap")[1]) / 100.0
                    cfg = replace(cfg, crypto_cap=cap)
                year = fold["year"]
                test = run(cfg, f"{year}-01-01", min(f"{year}-12-31", end))
                curves.append(test["curve"])
            stitched = _stitch(curves)
            sm = slice_metrics(stitched, stitched.index.min(), end, min_days) if len(stitched) else {}
            res["pass_walkforward"] = bool(sm and np.isfinite(sm.get("cagr", np.nan)) and sm["cagr"] > 0)
            res["pass_perturb"] = runs[parent].get("pass_perturb")
            res["pass_all"] = all(
                bool(res.get(k))
                for k in ("pass_beats_oos", "pass_maxdd", "pass_worst_year", "pass_perturb", "pass_walkforward")
            )
            res["wf_cagr"] = sm.get("cagr") if sm else float("nan")

    print("stress, stooq, monte carlo", flush=True)
    stress_rows_src = {}
    for uname, result in long_runs.items():
        stress_rows_src[uname] = {}
        for label, (a, b) in thresholds["stress_windows"].items():
            stress_rows_src[uname][label] = slice_metrics(result["curve"], a, b, min_year_days=20)

    stooq_symbols = {"SPY": "spy.us", "QQQ": "qqq.us", "GLD": "gld.us", "AAPL": "aapl.us", "MSFT": "msft.us"}
    stooq = load_stooq(stooq_symbols, cache)
    nasdaq_stats = probe_nasdaq(list(stooq_symbols))
    stooq_stats = []
    for name, series in stooq.items():
        if not isinstance(series, pd.Series):
            stooq_stats.append({"name": name, "status": str(series)})
            continue
        y = panel[(name, "close")].dropna()
        y.index = pd.DatetimeIndex(y.index).tz_localize(None)
        both = pd.concat([y.rename("yahoo"), series.rename("stooq")], axis=1).dropna()
        if len(both) < 100:
            stooq_stats.append({"name": name, "status": "short_overlap", "n": int(len(both))})
            continue
        yr = both["yahoo"].pct_change().dropna()
        sr = both["stooq"].pct_change().dropna()
        aligned = pd.concat([yr.rename("y"), sr.rename("s")], axis=1).dropna()
        corr = float(aligned["y"].corr(aligned["s"])) if len(aligned) > 30 else float("nan")
        stooq_stats.append(
            {
                "name": name,
                "status": "ok",
                "n": int(len(aligned)),
                "corr_daily": corr,
                "start": str(aligned.index[0].date()),
                "end": str(aligned.index[-1].date()),
            }
        )

    mc = {k: _monte_carlo(runs[k]["trades"], seed) for k in ("control", "S", "SC")}
    block = {k: _block_bootstrap_dd(runs[k]["curve"], seed) for k in ("control", "S", "SC")}

    rows = []
    rows.append(_row("primary", "control", "正本v1.1.1 全市场规则在流动性名单上", control))
    for key in ("S", "SC"):
        rows.append(_row("primary", key, runs[key].get("detail", ""), runs[key]))
    for name, res in {**cash_runs, **cap_runs, **deriv_runs}.items():
        rows.append(_row("variant", name, name, res))
    for name, res in long_runs.items():
        rows.append(_row("long", name, "1998起", res))
    for uname, blocks in stress_rows_src.items():
        for label, res in blocks.items():
            rows.append(_row("stress", f"{uname}_{label}", label, res))
    for name, res in perturb.items():
        rows.append(_row("perturbation", name, name, res))
    for label, info in wf.items():
        sm = info["metrics"] or {}
        sm = dict(sm)
        sm["detail"] = json.dumps(info["folds"], ensure_ascii=False, default=str)
        rows.append(_row("walkforward", label, "3年训练/1年测试", sm, {"pass_walkforward": info["positive"]}))
    for item in nasdaq_stats:
        rows.append(
            {
                "group": "nasdaq",
                "candidate": item["name"],
                "spec": item.get("status"),
                "detail": json.dumps(item, ensure_ascii=False),
                "n_days": item.get("n"),
                "start": item.get("start", ""),
                "end": item.get("end", ""),
            }
        )
    for item in stooq_stats:
        rows.append(
            {
                "group": "stooq",
                "candidate": item["name"],
                "spec": item.get("status"),
                "detail": json.dumps(item, ensure_ascii=False),
                "cagr": item.get("corr_daily"),
                "n_days": item.get("n"),
                "start": item.get("start", ""),
                "end": item.get("end", ""),
            }
        )
    for name, info in mc.items():
        rows.append(
            {
                "group": "montecarlo",
                "candidate": name,
                "spec": "trade-order bootstrap",
                "detail": json.dumps({**info, "block": block.get(name, {})}, ensure_ascii=False),
                "max_dd": info.get("maxdd_p50"),
                "cagr": info.get("p_year_loss_gt_30"),
                "n_trades": info.get("n_trades"),
            }
        )
    ref = thresholds["round1_reference_yahoo_2019"]
    for name, nums in ref.items():
        rows.append(
            {
                "group": "round1_reference",
                "candidate": name,
                "spec": "已发表的第一轮数字，不是这次重跑",
                "cagr": nums["ann"],
                "max_dd": nums["maxdd"],
                "oos_cagr": nums["oos_2024"],
            }
        )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "all.csv", index=False)
    meta = {
        "threshold_sha256": thresholds["_sha256"],
        "n_names": pack.n,
        "names": pack.names,
        "start": str(pack.dates[0].date()),
        "end": end,
        "seed": seed,
        "control_oos_cagr": control_oos,
        "note": "trading-system PR #13 was not accessible; rules reimplemented from the brief and the v1.1.1 engine.",
        "universe_counts": {
            "stocks": len([s for s in pack.names if s in STOCKS]),
            "commodity": len([s for s in pack.names if s in COMMODITY]),
            "crypto": len([s for s in pack.names if s in CRYPTO]),
            "levered_map": LEV_MAP,
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (out_dir / "walkforward.json").write_text(json.dumps({k: {"positive": v["positive"], "metrics": _jsonable(v["metrics"]), "folds": v["folds"]} for k, v in wf.items()}, ensure_ascii=False, indent=2))

    charts = _charts(out_dir / "charts", runs, long_runs, cap_runs, deriv_runs, wf, end, thresholds)
    summary = _summary_zh(
        runs, cash_runs, cap_runs, deriv_runs, long_runs, stress_rows_src, perturb, wf, mc, block, stooq_stats, nasdaq_stats, meta, thresholds
    )
    (out_dir / "summary_zh.md").write_text(summary)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "summary_zh.md").write_text(summary)
    for p in charts:
        target = ART / p.name
        target.write_bytes(p.read_bytes())
    print("done", out_dir, flush=True)
    return {"rows": frame, "runs": runs, "meta": meta}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "curve"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _pct(x):
    if x is None or not isinstance(x, (int, float, np.floating)) or not np.isfinite(x):
        return "—"
    return f"{float(x)*100:.1f}%"


def _num(x):
    if x is None or not isinstance(x, (int, float, np.floating)) or not np.isfinite(x):
        return "—"
    return f"{float(x):.2f}"


def _charts(folder, runs, long_runs, cap_runs, deriv_runs, wf, end, thresholds):
    folder = Path(folder)
    start = thresholds["primary_start"]
    oos = thresholds["oos_start"]
    period = f"{start} 至 {end}"
    paths = []

    def save(name, title, ylabel, series, baseline=0):
        path = folder / name
        line_chart(path, title, ylabel, series, baseline=baseline)
        paths.append(path)
        return path

    def nav(res):
        return res["curve"].dropna()

    save(
        "01_净值_主样本.png",
        f"因子网格第二轮｜主样本净值｜{period}\n正本 v1.1.1：30周均线、最多3只、现金30%、目标波动16%。S/SC：三周确认、20日动量、目标波动25%、现金0%",
        "净值（起点=1）",
        [
            (f"正本 v1.1.1｜{period}｜波动16% 现金30%", nav(runs["control"])),
            (f"S 股票｜{period}｜动量20日 波动25% 现金0%", nav(runs["S"])),
            (f"SC 股票+加密｜{period}｜同S，另加三种币", nav(runs["SC"])),
        ],
    )
    save(
        "02_回撤_主样本.png",
        f"因子网格第二轮｜主样本回撤｜{period}\n黑线是正本 v1.1.1。回撤从各自净值高点计算。样本外从 {oos} 起",
        "回撤",
        [
            (f"正本 v1.1.1｜{period}｜波动16% 现金30%", _drawdown(nav(runs["control"]))),
            (f"S 股票｜{period}｜动量20日 波动25% 现金0%", _drawdown(nav(runs["S"]))),
            (f"SC 股票+加密｜{period}｜同S，另加三种币", _drawdown(nav(runs["SC"]))),
        ],
    )
    save(
        "03_净值_长历史.png",
        f"因子网格第二轮｜长历史净值｜{thresholds['long_start']} 至 {end}\n股票从1998年起，币从上市日起。名单是当前流动性品种，有幸存者偏差",
        "净值（起点=1）",
        [
            (f"正本 v1.1.1｜{thresholds['long_start']}起｜30周均线 现金30%", nav(long_runs["control"])),
            (f"S 股票｜{thresholds['long_start']}起｜动量20日 现金0%", nav(long_runs["S"])),
            (f"SC 股票+加密｜{thresholds['long_start']}起｜同S", nav(long_runs["SC"])),
        ],
    )
    save(
        "04_加密仓位上限.png",
        f"因子网格第二轮｜加密仓位上限｜{period}\n黑线是上限0%（不买币）。其余规则同 SC：动量20日、目标波动25%、现金0%",
        "净值（起点=1）",
        [
            (f"加密上限 0%｜{period}｜不持有币", nav(cap_runs["SC_cryptocap0"])),
            (f"加密上限 20%｜{period}｜其余规则同 SC", nav(cap_runs["SC_cryptocap20"])),
            (f"加密上限 40%｜{period}｜其余规则同 SC", nav(cap_runs["SC_cryptocap40"])),
        ],
        baseline=0,
    )
    s_lev_days = int(deriv_runs["S_lev"].get("lev_days") or 0)
    sc_lev_days = int(deriv_runs["SC_lev"].get("lev_days") or 0)
    save(
        "05_衍生品_股票.png",
        f"因子网格第二轮｜杠杆ETF与长期看涨｜{period}\n橙线是正本换成 UGL/AGQ/UCO/TQQQ/SOXL。S 替换 {s_lev_days} 天，SC 替换 {sc_lev_days} 天，故不另画。蓝线是近似",
        "净值（起点=1）",
        [
            (f"正本 v1.1.1｜{period}｜无衍生品", nav(runs["control"])),
            (f"正本换杠杆ETF｜{period}｜标的与杠杆ETF都在上升段才替换", nav(deriv_runs["control_lev"])),
            (f"S 长期看涨｜{period}｜德尔塔0.7 约一年 半年展期（近似）", nav(deriv_runs["S_leap"])),
        ],
    )
    save(
        "06_牛市价差.png",
        f"因子网格第二轮｜牛市看涨价差（近似）｜{period}\n买德尔塔0.7、卖德尔塔0.4，约180天，90天展期。不是真实期权成交价",
        "净值（起点=1）",
        [
            (f"正本 v1.1.1｜{period}｜买正股或商品ETF", nav(runs["control"])),
            (f"S 牛市价差｜{period}｜布莱克-舒尔斯近似", nav(deriv_runs["S_spread"])),
            (f"SC 牛市价差｜{period}｜布莱克-舒尔斯近似", nav(deriv_runs["SC_spread"])),
        ],
    )
    save(
        "07_走步样本外.png",
        "因子网格第二轮｜走步样本外净值｜3年训练、1年测试后拼接\n训练网格：动量16/20/24、波动20/25/30%、均线24/30/36。取夏普最高且训练回撤和最差年不低于-30%的一组",
        "净值（起点=1）",
        [
            ("正本 v1.1.1｜参数固定｜与走步同一时段", nav(long_runs["control"]).loc[lambda s: s.index >= wf["S"]["stitched"].index.min()] if len(wf["S"]["stitched"]) else nav(runs["control"])),
            ("S 走步拼接｜3年训练 1年测试｜动量/波动/均线网格", wf["S"]["stitched"]),
            ("SC 走步拼接｜3年训练 1年测试｜同一网格", wf["SC"]["stitched"]),
        ],
    )
    return paths


def _summary_zh(runs, cash_runs, cap_runs, deriv_runs, long_runs, stress, perturb, wf, mc, block, stooq_stats, nasdaq_stats, meta, thresholds):
    def line(name, res):
        return (
            f"| {name} | {_pct(res.get('cagr'))} | {_pct(res.get('maxdd'))} | "
            f"{_pct(res.get('worst_year'))} ({res.get('worst_year_label') or '—'}) | "
            f"{_pct(res.get('oos_cagr'))} | {_num(res.get('sharpe'))} | {res.get('n_trades', '—')} |"
        )

    def gate(res):
        def mark(v):
            if v is True:
                return "过"
            if v is False:
                return "不过"
            return "—"
        return (
            f"样本外超过正本 {mark(res.get('pass_beats_oos'))}；"
            f"最大回撤 {mark(res.get('pass_maxdd'))}；"
            f"最差年 {mark(res.get('pass_worst_year'))}；"
            f"参数扰动 {mark(res.get('pass_perturb'))}；"
            f"走步为正 {mark(res.get('pass_walkforward'))}；"
            f"五项一起 {mark(res.get('pass_all'))}"
        )

    passed = [k for k in ("S", "SC") if runs[k].get("pass_all")]
    variant_pass = [k for k, v in {**cash_runs, **cap_runs, **deriv_runs}.items() if v.get("pass_all")]
    lev_long = long_runs.get("control_lev", {})
    if passed:
        rec = (
            "股票或股票加密候选通过了五道门槛："
            + "、".join(passed)
            + "。名单是当前大市值品种，早年没有退市股，数字偏乐观，不能直接当成可以上实盘的证据。"
        )
    elif variant_pass == ["control_lev"]:
        rec = (
            "股票候选 S、股票加密候选 SC，以及现金下限、加密上限、期权近似，都没有同时通过五道门槛。"
            f"只有正本换成杠杆 ETF 这一条在主样本上五项都过：年化 {_pct(deriv_runs['control_lev'].get('cagr'))}，"
            f"最大回撤 {_pct(deriv_runs['control_lev'].get('maxdd'))}，最差年 {_pct(deriv_runs['control_lev'].get('worst_year'))}，"
            f"2024 年起年化 {_pct(deriv_runs['control_lev'].get('oos_cagr'))}。"
            f"同一条规则从 1998 年起的年化是 {_pct(lev_long.get('cagr'))}，最大回撤 {_pct(lev_long.get('maxdd'))}。"
            "这还不是把正本改成杠杆的理由：名单没有退市股，杠杆 ETF 有波动损耗，走步没有重新选参。"
            "建议继续用不加杠杆的正本。若要认真考虑杠杆替换，先在含退市股的全市场样本上再跑一遍。"
        )
    elif variant_pass:
        rec = (
            "S 和 SC 本身没有通过。通过五道门槛的是："
            + "、".join(variant_pass)
            + "。名单是当前大市值品种，早年没有退市股，不能直接上实盘。"
        )
    else:
        rec = (
            "股票候选和股票加密候选都没有同时通过五道门槛。"
            "建议维持正本 v1.1.1（买正股和商品 ETF，不加重杠杆、不换期权）。"
            "第一轮较高的样本外收益，在这次重跑里没有保持住。"
        )

    stress_lines = []
    for uname in ("control", "S", "SC", "control_lev"):
        bits = []
        for label, res in stress[uname].items():
            bits.append(f"{label} 收益 {_pct(res.get('cagr'))}、区间回撤 {_pct(res.get('maxdd'))}")
        stress_lines.append(f"- {uname}：" + "；".join(bits))

    mc_lines = []
    for name in ("control", "S", "SC"):
        info = mc[name]
        extra = block.get(name, {})
        mc_lines.append(
            f"- {name}：成交 {info.get('n_trades', 0)} 笔。重排成交顺序 2000 次后，回撤中位数 {_pct(info.get('maxdd_p50'))}，"
            f"较差的5%分位 {_pct(info.get('maxdd_p05'))}。任一年亏损超过30%的概率（按年计数）{_pct(info.get('p_year_loss_gt_30'))}，"
            f"一条路径里至少出现一次这种年份的概率 {_pct(info.get('p_sim_has_year_loss_gt_30'))}。"
            f"日收益按21天一块重抽的回撤中位数 {_pct(extra.get('maxdd_p50'))}。"
        )

    stooq_lines = []
    stooq_ok = 0
    for item in stooq_stats:
        if item.get("status") == "ok":
            stooq_ok += 1
            stooq_lines.append(
                f"- {item['name']}：与雅虎日收益相关系数 {item['corr_daily']:.3f}，重叠 {item['n']} 天（{item['start']} 至 {item['end']}）。"
            )
        else:
            stooq_lines.append(f"- Stooq {item['name']}：没有对上（{item.get('status')}）。")
    for item in nasdaq_stats:
        stooq_lines.append(f"- 纳斯达克公开接口 {item['name']}：{item.get('status')}，记录 {item.get('n', 0)} 条。")
    if stooq_ok:
        stooq_close = "上面相关系数接近 1 的几只，只说明这两个免费源的收盘价大体一致。这不是全名单核对。"
    else:
        stooq_close = "Stooq 被网站的浏览器校验拦住，纳斯达克公开历史接口对这五只也没有返回日线。这次没有做成第二个免费源的交叉核对，不能用它来确认雅虎价格。"

    cash_lines = "\n".join(line(k, v) for k, v in cash_runs.items())
    cap_lines = "\n".join(line(k, v) for k, v in cap_runs.items())
    deriv_lines = "\n".join(line(k, v) for k, v in deriv_runs.items())
    deriv_notes = []
    for k, v in deriv_runs.items():
        days = f" 杠杆替换 {int(v.get('lev_days') or 0)} 天。" if k.endswith("lev") else ""
        deriv_notes.append(f"- {k}：{gate(v)}。{days}")
    long_lines = "\n".join(line(k, v) for k, v in long_runs.items())

    text = f"""# 因子网格第二轮：稳健性和衍生品

这次没有读到交易系统仓库的第13号拉取请求（仓库从这里访问不到）。规则按任务说明，以及正本 v1.1.1 引擎的做法重写：周线收盘在抬升的30周均线之上才算上升段，最多持有3只，黄金、白银、原油优先，现金至少30%，斐波那契阶段系数，组合目标波动16%，离开上升段先减半，按历史盈利情况排序，允许零股，不加杠杆。

股票池不是全市场。用的是固定的大市值股票、宽基和行业ETF，加上三种商品、三种加密货币，以及点名的杠杆ETF。退市股不在名单里，所以长历史会偏乐观。价格来自雅虎复权日线，缓存在本地且不进版本库。随机种子 {meta['seed']}。门槛文件摘要 {meta['threshold_sha256'][:12]}。样本从 {meta['start']} 到 {meta['end']}，主比较从 2019-01-02 起，样本外从 2024-01-01 起。

期权没有用历史成交价。长期看涨和牛市价差都是布莱克-舒尔斯公式，隐含波动率取63日已实现波动再乘1.20，利率2%，买入时每边再扣权利金或价差的一截成本。这是近似，不是交易所报价。

## 主样本

| 候选 | 年化 | 最大回撤 | 最差完整年 | 2024年起年化 | 夏普 | 成交笔数 |
| --- | --- | --- | --- | --- | --- | --- |
{line("正本", runs["control"])}
{line("S 股票", runs["S"])}
{line("SC 股票+加密", runs["SC"])}

第一轮已经发表、这次没有重跑的数字：正本 13.3% / 回撤 -16.7% / 样本外 20.8%；S 20.0% / -22.9% / 41.0%；SC 27.9% / -22.7% / 54.6%。两边名单不同，不要当成同一条曲线。

S：{gate(runs["S"])}

SC：{gate(runs["SC"])}

## 现金下限和加密上限

| 候选 | 年化 | 最大回撤 | 最差完整年 | 2024年起年化 | 夏普 | 成交笔数 |
| --- | --- | --- | --- | --- | --- | --- |
{cash_lines}
{cap_lines}

加密上限0%就是不买币。如果它和上限40%的样本外差距很大，说明 SC 的超额主要来自币，而不是股票规则。

## 衍生品

杠杆ETF在它自己也处于上升段时，替换对应标的：黄金换UGL，白银换AGQ，原油换UCO，纳指换TQQQ，半导体换SOXL。S 和 SC 的买入名单不含这三种商品。若替换天数是 0，这两本书就没有用上杠杆ETF。正本会持有商品，所以另跑了一条正本换杠杆ETF。期权按上面的近似计价。

| 候选 | 年化 | 最大回撤 | 最差完整年 | 2024年起年化 | 夏普 | 成交笔数 |
| --- | --- | --- | --- | --- | --- | --- |
{deriv_lines}

{chr(10).join(deriv_notes)}

约束是最大回撤和最差完整年都不要差于 -30%。上表的「最大回撤」「最差年」两列就是这个检查。正本衍生品的参数扰动是在正本自己的旋钮上各上下 20%（目标波动 12.8% 与 19.2%，均线 24 与 36）。动量窗口不进入正本的排序，所以那两档与基准相同。正本衍生品的走步不重新选参，只把固定规则按年拼接。S 和 SC 的现金下限、加密上限和衍生品，参数扰动沿用对应股票书，没有在期权上再扫一遍。

## 长历史和几段已知的差年份

| 候选 | 年化 | 最大回撤 | 最差完整年 | 2024年起年化 | 夏普 | 成交笔数 |
| --- | --- | --- | --- | --- | --- | --- |
{long_lines}

差年份（区间收益是这段里的年化，不是整段总收益）：

{chr(10).join(stress_lines)}

长历史跌破 -30% 不会单独否决主样本门槛，但只要出现，就不该把主样本的通过当成能扛危机。

## 参数扰动和走步

扰动是一次只动一个旋钮：动量窗口、目标波动、均线长度，各上下20%。年化掉到基准的一半以下算崩溃。

S 扰动：{"过" if runs["S"].get("pass_perturb") else "不过"}。SC 扰动：{"过" if runs["SC"].get("pass_perturb") else "不过"}。

走步是3年训练、下一年测试。训练样本里只保留回撤和最差年都不差于 -30% 的参数，再取夏普最高的一组；没有合格组就退回夏普最高的一组。S 拼接后年化 {_pct((wf["S"]["metrics"] or {}).get("cagr"))}，{"为正" if wf["S"]["positive"] else "不为正"}。SC 拼接后年化 {_pct((wf["SC"]["metrics"] or {}).get("cagr"))}，{"为正" if wf["SC"]["positive"] else "不为正"}。

## 成交顺序重抽

{chr(10).join(mc_lines)}

这是把单笔盈亏重新排队，不是把整段净值重抽。重叠持仓时，它会低估真实回撤，所以旁边附了按21个交易日一块重抽日收益的回撤中位数。

## 第二个免费行情源

{chr(10).join(stooq_lines)}

{stooq_close}

## 若要改用真实期权价，公开标价（没有购买）

- Databento OPRA 历史：按未压缩体积从大约每 GB 0.04 美元起算，下载前可以用接口询价。标准套餐公开价每月 199 美元，含十多年汇总数据和近12个月的报价与成交；更早的逐笔报价仍按用量计。个人行情许可大约每终端 1.25 美元另计，机构许可从大约 2000 美元起。新户有 125 美元额度。上一轮这个账户用真实价差数据时，Databento 花费不超过 15.65 美元（批准上限 18 美元），那次只覆盖了有限的合约和两个时点。
- ORATS：自 2007 年起全市场每日行权价、含希腊值和理论价，公开的一次性历史包是 599 美元。实时接口个人使用大约每月 299 美元，15分钟延迟大约每月 199 美元。
- Cboe DataShop：没有公开价目表，按品种、日期和是否含希腊值计价。2026-09-29 有人用它的计算器得到：SPY 一年日频链含希腊值大约 240 美元，全市场一年大约 2100 美元，2012 年以来全市场含希腊值大约 8500 美元。标普指数期权的一分钟报价，大约一年 550 美元，满四年后封顶大约 2200 美元（不含计算项）。

这次没有下单买任何一份数据。

## 建议

{rec}

主要风险：幸存者偏差；加密货币只在美股交易日上计价，周末涨跌算进下一个美股交易日；期权是模型价，真实买卖价更差，上一轮用真实数据做过的牛市价差（年化 3.7%、回撤 -31.8%）并没有打赢买股票；杠杆ETF有波动损耗，波动率用的是滞后的指数加权，急跌时减仓偏慢。
"""
    return text
