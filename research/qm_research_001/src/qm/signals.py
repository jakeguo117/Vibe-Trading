"""Signal extraction: boolean trigger matrix -> ranked signal table."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build(kit, trig, style, d0, d1):
    sub = trig[d0:d1 + 1]
    dd, jj = np.nonzero(sub)
    dd = dd + d0
    if len(dd) == 0:
        return pd.DataFrame(columns=["d", "j", "stop", "atr", "k1", "k2", "k3"])
    p = dd - 1
    stop = kit.l[dd, jj].astype(float)
    atr = kit.atr14[p, jj].astype(float)
    ok = np.isfinite(stop) & np.isfinite(atr) & (atr > 0)
    dd, jj, p, stop, atr = dd[ok], jj[ok], p[ok], stop[ok], atr[ok]

    if style == "bo":
        k1 = kit.r63[p, jj].astype(float)
        k2 = kit.dv20[p, jj].astype(float)
    elif style == "ps":
        k1 = kit.r10[p, jj].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            k2 = kit.c[p, jj] / kit.sma10[p, jj] - 1.0
    elif style == "ep":
        v20 = np.full(len(dd), np.nan)
        for i in range(len(dd)):
            a = max(0, p[i] - 20)
            vv = kit.v[a:p[i], jj[i]]
            v20[i] = np.nanmean(vv) if len(vv) else np.nan
        k1 = kit.v[dd, jj].astype(float) / v20
        with np.errstate(invalid="ignore", divide="ignore"):
            k2 = kit.o[dd, jj] / kit.c[p, jj] - 1.0
    else:
        raise ValueError(style)

    out = pd.DataFrame({"d": dd, "j": jj, "stop": stop, "atr": atr,
                        "k1": k1, "k2": k2, "k3": jj})
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["k1", "k2"])
    return out
