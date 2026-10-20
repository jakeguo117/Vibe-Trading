"""A-line per-episode table (plan §8.1/§9.2): A is a weight engine, so a "trade" is a code's
continuous holding episode. Reports both the price return of that code and its contribution to
account NAV, which §9.2 requires to be kept apart.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from .adriver import A_VARIANTS, _load_D
from .metrics import trade_stats


def episodes(D, weights: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    C = D.C
    rows = []
    for name, g in weights.groupby("name"):
        g = g.sort_values("t")
        j = int(g.j.iloc[0])
        held = (g.w_close.fillna(0.0) > 1e-12).values
        t = g.t.values
        w = g.w_close.fillna(0.0).values
        i = 0
        while i < len(held):
            if not held[i]:
                i += 1
                continue
            k = i
            while k + 1 < len(held) and held[k + 1]:
                k += 1
            t0, t1 = int(t[i]), int(t[k])
            px0, px1 = C[t0, j], C[t1, j]
            rr = np.array([C[t[i2] + 1, j] / C[t[i2], j] - 1 if t[i2] + 1 < C.shape[0] else 0.0
                           for i2 in range(i, k + 1)])
            contrib = float(np.nansum(w[i:k + 1] * rr))
            # §9.3 wants price MAE/MFE, but D_LOCKED stores no high/low for the full universe
            # (mat/ has open/close/dv only), so the A line reports the CLOSE-ONLY basis and says so.
            cc = C[t0:t1 + 1, j]
            fin = cc[np.isfinite(cc)]
            mae = float(np.min(fin) / px0 - 1.0) if np.isfinite(px0) and len(fin) else np.nan
            mfe = float(np.max(fin) / px0 - 1.0) if np.isfinite(px0) and len(fin) else np.nan
            rows.append(dict(name=name, entry_t=t0, exit_t=t1, entry_date=D.cal[t0], exit_date=D.cal[t1],
                             days=t1 - t0 + 1, price_ret=float(px1 / px0 - 1.0) if np.isfinite(px0) and np.isfinite(px1) else np.nan,
                             nav_contribution=contrib,
                             avg_w=float(np.nanmean(w[i:k + 1])),
                             mae=mae, mfe=mfe, basis="close_only",
                             ma1=float(np.nanmin(cc[:2]) / px0 - 1.0) if len(cc) > 1 else np.nan,
                             ma5=float(np.nanmin(cc[:6]) / px0 - 1.0) if len(cc) > 1 else np.nan,
                             net_1d=float(cc[min(1, len(cc) - 1)] / px0 - 1.0),
                             net_5d=float(cc[min(5, len(cc) - 1)] / px0 - 1.0)))
            i = k + 1
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--backtest-dir", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.backtest_dir)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = _load_D(a.data_dir)
    allstats = {}
    frames = []
    for vid in A_VARIANTS:
        w = pd.read_csv(out / f"weights_{vid}.csv")
        e = pd.read_csv(out / f"events_{vid}.csv")
        ep = episodes(D, w, e)
        ep.insert(0, "variant", vid)
        frames.append(ep)
        st = trade_stats(ep.rename(columns={"net_ret": "nav_contribution"}).assign(
            net_ret=ep.nav_contribution))
        allstats[vid] = {"n_episodes": int(len(ep)), **st,
                         "mean_days": float(ep.days.mean()), "median_days": float(ep.days.median()),
                         "mean_price_ret": float(ep.price_ret.mean()),
                         "mean_nav_contribution": float(ep.nav_contribution.mean())}
        print(vid, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in allstats[vid].items()}))
    EP = pd.concat(frames, ignore_index=True)
    EP.to_csv(out / "a_episodes.csv", index=False, float_format="%.10g")
    (out / "a_episodes.json").write_text(json.dumps(allstats, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
